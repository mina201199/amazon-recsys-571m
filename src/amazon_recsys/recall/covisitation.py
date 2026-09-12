"""共現召回：「買了 A 的人也買了 B」。

共現統計依賴重複出現的互動配對，通道品質需與熱門基線比較。

## 核心工程問題：自連接的平方成本

商品配對是由「同一使用者買過的商品兩兩配對」產生的，成本是每位
使用者歷史長度的平方。全量資料平均每人 10.5 筆，看似無害，但分布
是冪律的：少數使用者有數百筆歷史，單獨一位就能貢獻數萬個配對。

三道防線，每一道都有明確理由：

1. `max_items_per_user` —— 只取每位使用者最近 N 筆。
   這同時解決成本與品質：一位買過 500 樣東西的使用者，他早期的
   購買與近期的興趣關聯很弱，全部納入反而是雜訊。
2. `window_days` —— 只統計近期互動。
   2015 年的共現關係對預測 2023 年的行為幫助有限。
3. `min_cooccurrence` —— 只出現一次的配對多半是巧合，不是訊號。

## 分數正規化

原始共現次數會讓熱門商品獨占所有鄰居清單（熱門商品和什麼都共現）。
因此除以兩個商品熱度的幾何平均（餘弦式正規化）：

    score(a, b) = cnt(a, b) / sqrt(pop(a) * pop(b))

這讓「相對於各自熱度而言，異常常一起出現」的配對浮上來。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from amazon_recsys.recall.base import PAD

SECONDS_PER_DAY = 86_400


@dataclass
class CoVisitationRecall:
    """基於商品共現的召回通道。"""

    window_days: int | None = 730       # 近兩年
    max_items_per_user: int = 50        # 見模組說明的防線 1
    top_n_neighbours: int = 100         # 每個商品保留的鄰居數
    min_cooccurrence: int = 2           # 濾掉只共現一次的巧合
    name: str = "covisitation"
    _table: str = "covis_neighbours"
    _con: object | None = None

    def fit(self, con, src: str, cutoff: int) -> None:
        self._con = con
        lo = cutoff - self.window_days * SECONDS_PER_DAY if self.window_days else None
        where = f"ts < {cutoff}" + (f" AND ts >= {lo}" if lo is not None else "")

        # 防線 1+2：限定時間視窗，並只取每位使用者最近 N 筆
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE covis_input AS
            SELECT user_idx, item_idx FROM (
                SELECT user_idx, item_idx,
                       row_number() OVER (PARTITION BY user_idx ORDER BY ts DESC) AS rn
                FROM {src} WHERE {where}
            ) WHERE rn <= {self.max_items_per_user}
        """)

        con.execute("""
            CREATE OR REPLACE TEMP TABLE covis_pop AS
            SELECT item_idx, count(*) AS pop FROM covis_input GROUP BY item_idx
        """)

        # 自連接產生配對。a.item_idx < b.item_idx 只算一半，
        # 之後再對稱展開——省一半計算量。
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE covis_pairs AS
            SELECT a.item_idx AS lo, b.item_idx AS hi, count(*) AS cnt
            FROM covis_input a
            JOIN covis_input b
              ON a.user_idx = b.user_idx AND a.item_idx < b.item_idx
            GROUP BY 1, 2
            HAVING count(*) >= {self.min_cooccurrence}
        """)

        # 對稱展開 + 正規化 + 每個商品保留前 N 個鄰居
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {self._table} AS
            SELECT src, dst, score FROM (
                SELECT src, dst, score,
                       row_number() OVER (PARTITION BY src ORDER BY score DESC, dst) AS rn
                FROM (
                    SELECT p.lo AS src, p.hi AS dst,
                           p.cnt / sqrt(CAST(ps.pop AS DOUBLE) * pd.pop) AS score
                    FROM covis_pairs p
                    JOIN covis_pop ps ON ps.item_idx = p.lo
                    JOIN covis_pop pd ON pd.item_idx = p.hi
                    UNION ALL
                    SELECT p.hi AS src, p.lo AS dst,
                           p.cnt / sqrt(CAST(ps.pop AS DOUBLE) * pd.pop) AS score
                    FROM covis_pairs p
                    JOIN covis_pop ps ON ps.item_idx = p.lo
                    JOIN covis_pop pd ON pd.item_idx = p.hi
                )
            ) WHERE rn <= {self.top_n_neighbours}
        """)
        con.execute("DROP TABLE covis_pairs")

    def stats(self) -> dict[str, int]:
        """鄰居表的規模，用於報告與偵錯。"""
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        n_edges, n_items = self._con.execute(
            f"SELECT count(*), count(DISTINCT src) FROM {self._table}"
        ).fetchone()
        return {"edges": n_edges, "items_with_neighbours": n_items}

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        """對每位使用者，把歷史商品的鄰居分數加總後取前 k 名。

        用 SQL 做批次運算而非 Python 迴圈：使用者數可能達數百萬，
        逐一查表會慢到不可用。
        """
        if self._con is None:
            raise RuntimeError("尚未呼叫 fit()")
        con = self._con

        out = np.full((len(histories), k), PAD, dtype=np.int64)

        # 攤平成兩個平行陣列，再以 Arrow 批次載入。
        #
        # 不用 executemany 逐列 INSERT：DuckDB 沒有批次路徑，
        # 兩萬位使用者的歷史約 46 萬列，逐列寫入實測要十幾分鐘，
        # 而批次載入是秒級。這裡的成本原本佔了整個評估的絕大部分。
        row_ids: list[int] = []
        items: list[int] = []
        for i, h in enumerate(histories):
            row_ids.extend([i] * len(h))
            items.extend(int(x) for x in h)
        if not row_ids:
            return out

        query_hist = pa.table({
            "row_id": pa.array(row_ids, pa.int32()),
            "item_idx": pa.array(items, pa.int32()),
        })
        con.register("query_hist_arrow", query_hist)
        con.execute(
            "CREATE OR REPLACE TEMP TABLE query_hist AS SELECT * FROM query_hist_arrow"
        )
        con.unregister("query_hist_arrow")

        result = con.execute(f"""
            SELECT row_id, dst FROM (
                SELECT q.row_id, n.dst, sum(n.score) AS score,
                       row_number() OVER (
                           PARTITION BY q.row_id ORDER BY sum(n.score) DESC, n.dst
                       ) AS rn
                FROM query_hist q
                JOIN {self._table} n ON n.src = q.item_idx
                -- 排除使用者已經互動過的商品
                WHERE NOT EXISTS (
                    SELECT 1 FROM query_hist s
                    WHERE s.row_id = q.row_id AND s.item_idx = n.dst
                )
                GROUP BY q.row_id, n.dst
            ) WHERE rn <= {k}
            ORDER BY row_id, rn
        """).fetchall()

        # 結果已依 row_id, rn 排序，依序寫入各列即可
        pos: dict[int, int] = {}
        for row_id, dst in result:
            p = pos.get(row_id, 0)
            if p < k:
                out[row_id, p] = dst
                pos[row_id] = p + 1
        return out
