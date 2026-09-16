"""類別熱門召回：把最強的單路通道限制在使用者買過的類別裡。

驗證集上全域熱門是最強的單路通道（Recall@500 = 0.0253），而共現只有
0.0063、ALS 0.0125、內容 0.0103。但全域熱門只用了約 507 件商品，
而且不管使用者買什麼都推同一份清單——它的分數幾乎全部來自
「暢銷品本來就有人買」，不含任何個人化訊號。

這一路只改一件事：熱度改成在使用者買過的類別內計算。`category_idx`
本來就在互動表裡，不需要 metadata、不需要訓練、也不需要新的資料管線。

## 分數

    score(商品) = Σ_類別  使用者在該類別的互動數 / (rank_constant + 類別內熱度名次)

用名次而非原始次數，理由與融合層用 RRF 相同：不同類別的熱度規模差很多
（Books 與 Subscription_Boxes 差幾個數量級），原始次數相加會讓大類別
輾壓小類別；名次把每個類別放到同一個尺度上。

分子是使用者在該類別的互動數：買過 10 本書、1 個工具的人，書的候選
應該比工具多。這是刻意不做正規化的——類別偏好本身就是強度訊號。

## 兩個時間窗

熱度只取 `window_days` 內的互動（與全域熱門一致，理由相同：十年前的
暢銷書對預測近期行為沒有幫助）。但**商品到類別的對應取整個 cutoff
之前的資料**——使用者的歷史商品可能很舊，若也套熱度窗，舊商品會查不到
類別，使用者就整個掉出這一路。兩者都嚴格早於 cutoff，不構成洩漏。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from amazon_recsys.recall.base import PAD

SECONDS_PER_DAY = 86_400


@dataclass
class CategoryPopularityRecall:
    """依使用者歷史類別加權的熱門商品召回。"""

    window_days: int | None = 90
    per_category: int = 2_000      # 每個類別保留的候選數
    rank_constant: float = 60.0    # 與融合層的 RRF 同一個常數
    name: str = "category_popularity"

    _con: object | None = None
    _pool: str = "catpop_pool"
    _item_cat: str = "catpop_item_cat"

    def fit(self, con, src: str, cutoff: int) -> None:
        self._con = con
        lo = cutoff - self.window_days * SECONDS_PER_DAY if self.window_days else None
        where = f"ts < {cutoff}" + (f" AND ts >= {lo}" if lo is not None else "")

        # 商品到類別：取整個 cutoff 之前，讓舊歷史商品也查得到類別
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {self._item_cat} AS
            SELECT DISTINCT item_idx, category_idx FROM {src} WHERE ts < {cutoff}
        """)

        # 每個類別的熱門商品與類別內名次；item_idx 作為決勝鍵以確保可重現
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {self._pool} AS
            SELECT category_idx, item_idx, pop, rn FROM (
                SELECT category_idx, item_idx, pop,
                       row_number() OVER (PARTITION BY category_idx
                                          ORDER BY pop DESC, item_idx) AS rn
                FROM (
                    SELECT category_idx, item_idx, count(*) AS pop
                    FROM {src} WHERE {where}
                    GROUP BY category_idx, item_idx
                )
            ) WHERE rn <= {self.per_category}
        """)

    def stats(self) -> dict[str, int]:
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        n_rows, n_cats = self._con.execute(
            f"SELECT count(*), count(DISTINCT category_idx) FROM {self._pool}"
        ).fetchone()
        return {"pool_rows": n_rows, "categories": n_cats}

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        con = self._con
        out = np.full((len(histories), k), PAD, dtype=np.int64)

        row_ids = [i for i, h in enumerate(histories) for _ in h]
        items = [int(x) for h in histories for x in h]
        if not row_ids:
            return out

        con.register("catpop_hist_arrow", pa.table({
            "row_id": pa.array(row_ids, pa.int64()),
            "item_idx": pa.array(items, pa.int64()),
        }))
        try:
            rows = con.execute(f"""
                WITH user_cats AS (
                    SELECT h.row_id, c.category_idx, count(*) AS w
                    FROM catpop_hist_arrow h JOIN {self._item_cat} c USING (item_idx)
                    GROUP BY 1, 2
                )
                SELECT row_id, item_idx FROM (
                    SELECT u.row_id, p.item_idx,
                           row_number() OVER (
                               PARTITION BY u.row_id
                               ORDER BY sum(u.w / ({self.rank_constant} + p.rn)) DESC,
                                        p.item_idx
                           ) AS rn
                    FROM user_cats u JOIN {self._pool} p USING (category_idx)
                    -- 排除使用者已經互動過的商品
                    WHERE NOT EXISTS (
                        SELECT 1 FROM catpop_hist_arrow s
                        WHERE s.row_id = u.row_id AND s.item_idx = p.item_idx
                    )
                    GROUP BY u.row_id, p.item_idx
                ) WHERE rn <= {k}
                ORDER BY row_id, rn
            """).fetchall()
        finally:
            con.unregister("catpop_hist_arrow")

        # 結果已依 row_id, rn 排序，依序寫入各列即可
        pos: dict[int, int] = {}
        for row_id, item_idx in rows:
            p = pos.get(row_id, 0)
            if p < k:
                out[row_id, p] = item_idx
                pos[row_id] = p + 1
        return out
