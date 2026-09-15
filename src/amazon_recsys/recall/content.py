"""內容式召回：靠商品屬性而非互動紀錄。

現有三路都需要商品**已經被買過**才找得到它：

    熱門    需要熱度     -> 新商品沒有
    共現    需要共現紀錄 -> 新商品沒有
    ALS     需要潛在向量 -> 新商品沒有

而實測顯示答案裡有 13.5% 是切分點前從未出現過的商品。
**這一路是唯一碰得到它們的方法**：一件沒人買過的商品依然有品牌、
類別與價格。

## 訊號的優先順序

以電商的實務經驗排序，強度由高至低：

1. **同品牌** —— 買過某品牌的人再買同品牌，是最強的內容訊號，
   而且對全新商品完全適用。
2. **同類別 + 相近價位** —— 較弱但覆蓋面廣，價位能濾掉同類別中
   明顯不符消費水準的商品。

## 刻意不用的訊號

metadata 的 `average_rating` 與 `rating_number` 是 2023-09 的彙總值，
含切分點之後的評論，用來排序等於偷看未來。品質訊號一律改由互動表在
切分點前重算（見 `fit` 中的熱度表）。

## 冷啟動的取捨

若候選一律按熱度排序，冷啟動商品永遠排不進來——那就失去這一路的意義。
因此 `cold_slots` 會在每個品牌的候選中**保留固定名額給零互動商品**，
用參數把這個取捨顯式化，而不是讓它被熱度悄悄吃掉。

注意：**類別池沒有這個保留機制**，它純按熱度取前 N。在 4819 萬件商品
的真實目錄下，單一類別動輒數百萬件，`per_category` 取到的幾乎必然全是
熱門商品。也就是說**冷啟動商品實際上只能經由品牌池進入候選**。
這是刻意的取捨：類別 + 冷啟動的組合訊號太弱，放進來多半是雜訊。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from amazon_recsys.recall.base import PAD


@dataclass
class ContentRecall:
    """基於商品靜態屬性（品牌、類別、價格）的召回通道。"""

    items_table: str                       # 商品屬性表的 read_parquet(...) 字串
    per_store: int = 200                   # 每個品牌保留的候選數
    per_category: int = 400                # 每個類別保留的候選數
    cold_slots: int = 50                   # 每個品牌保留給零互動商品的名額
    price_tolerance: float = 0.5           # 價格帶：使用者中位價的 ±50%
    name: str = "content"

    _con: object | None = None
    _pop: str = "content_item_pop"
    _items: str = "content_items"

    def fit(self, con, src: str, cutoff: int) -> None:
        self._con = con

        # 切分點前的熱度——不取自 metadata 的 rating_number（那含未來資料）
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {self._pop} AS
            SELECT item_idx, count(*) AS pop
            FROM {src} WHERE ts < {cutoff}
            GROUP BY item_idx
        """)

        # 屬性表併上熱度；沒有任何切分點前互動的商品 pop 為 0（即冷啟動）
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {self._items} AS
            SELECT i.item_idx, i.main_category_idx, i.store_idx, i.price,
                   coalesce(p.pop, 0) AS pop
            FROM {self.items_table} i
            LEFT JOIN {self._pop} p USING (item_idx)
        """)

        # 每個品牌的候選：熱門名額 + 保留給冷啟動的名額。
        # 分開取而非統一排序，否則冷啟動商品永遠排不進來。
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE content_store_pool AS
            SELECT store_idx, item_idx, price, pop FROM (
                SELECT store_idx, item_idx, price, pop,
                       row_number() OVER (PARTITION BY store_idx
                                          ORDER BY pop DESC, item_idx) AS warm_rn,
                       row_number() OVER (PARTITION BY store_idx, pop = 0
                                          ORDER BY item_idx) AS cold_rn
                FROM {self._items} WHERE store_idx IS NOT NULL
            )
            WHERE warm_rn <= {self.per_store}
               OR (pop = 0 AND cold_rn <= {self.cold_slots})
        """)

        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE content_cat_pool AS
            SELECT main_category_idx, item_idx, price, pop FROM (
                SELECT main_category_idx, item_idx, price, pop,
                       row_number() OVER (PARTITION BY main_category_idx
                                          ORDER BY pop DESC, item_idx) AS rn
                FROM {self._items} WHERE main_category_idx IS NOT NULL
            ) WHERE rn <= {self.per_category}
        """)

    def stats(self) -> dict[str, int]:
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        n_items, n_cold = self._con.execute(f"""
            SELECT count(*), count(*) FILTER (WHERE pop = 0) FROM {self._items}
        """).fetchone()
        n_store, n_cat = self._con.execute("""
            SELECT (SELECT count(*) FROM content_store_pool),
                   (SELECT count(*) FROM content_cat_pool)
        """).fetchone()
        return {
            "items_with_attributes": n_items,
            "cold_items": n_cold,
            "store_pool_rows": n_store,
            "category_pool_rows": n_cat,
        }

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        con = self._con
        out = np.full((len(histories), k), PAD, dtype=np.int64)

        row_ids = [i for i, h in enumerate(histories) for _ in h]
        items = [int(x) for h in histories for x in h]
        if not row_ids:
            return out

        con.register("content_hist_arrow", pa.table({
            "row_id": pa.array(row_ids, pa.int64()),
            "item_idx": pa.array(items, pa.int64()),
        }))
        try:
            # 使用者輪廓：買過的品牌、類別權重、價格中位數
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE content_profile AS
                SELECT h.row_id, i.store_idx, i.main_category_idx, i.price
                FROM content_hist_arrow h JOIN {self._items} i USING (item_idx)
            """)
            con.execute("""
                CREATE OR REPLACE TEMP TABLE content_user_price AS
                SELECT row_id, median(price) AS med_price
                FROM content_profile WHERE price IS NOT NULL GROUP BY row_id
            """)

            rows = con.execute(f"""
                WITH user_stores AS (
                    SELECT DISTINCT row_id, store_idx FROM content_profile
                    WHERE store_idx IS NOT NULL
                ),
                user_cats AS (
                    SELECT row_id, main_category_idx, count(*) AS w
                    FROM content_profile WHERE main_category_idx IS NOT NULL
                    GROUP BY 1, 2
                ),
                -- 同品牌：最強的內容訊號，權重 2.0
                by_store AS (
                    SELECT u.row_id, p.item_idx, 2.0 AS sig, p.pop, p.price
                    FROM user_stores u JOIN content_store_pool p USING (store_idx)
                ),
                -- 同類別：較弱但覆蓋廣，權重依使用者在該類別的購買次數
                by_cat AS (
                    SELECT u.row_id, p.item_idx,
                           1.0 * u.w / (u.w + 3) AS sig, p.pop, p.price
                    FROM user_cats u JOIN content_cat_pool p USING (main_category_idx)
                ),
                pooled AS (SELECT * FROM by_store UNION ALL SELECT * FROM by_cat),
                scored AS (
                    SELECT c.row_id, c.item_idx,
                           max(c.sig) + CASE
                               WHEN up.med_price IS NULL OR c.price IS NULL THEN 0
                               WHEN abs(c.price - up.med_price)
                                    <= up.med_price * {self.price_tolerance} THEN 0.5
                               ELSE 0 END AS score,
                           max(c.pop) AS pop
                    FROM pooled c
                    LEFT JOIN content_user_price up USING (row_id)
                    WHERE NOT EXISTS (
                        SELECT 1 FROM content_hist_arrow s
                        WHERE s.row_id = c.row_id AND s.item_idx = c.item_idx)
                    GROUP BY c.row_id, c.item_idx, up.med_price, c.price
                )
                SELECT row_id, item_idx FROM (
                    SELECT row_id, item_idx,
                           row_number() OVER (PARTITION BY row_id
                                              ORDER BY score DESC, pop DESC, item_idx) AS rn
                    FROM scored
                ) WHERE rn <= {k}
                ORDER BY row_id, rn
            """).fetchall()
        finally:
            con.unregister("content_hist_arrow")

        pos: dict[int, int] = {}
        for row_id, item_idx in rows:
            p = pos.get(row_id, 0)
            if p < k:
                out[row_id, p] = item_idx
                pos[row_id] = p + 1
        return out
