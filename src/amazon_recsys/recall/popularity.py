"""熱門商品召回 —— 專案的基準線。

這一路存在的理由不是它強，而是它是**必須被打敗的對手**。
推薦系統有個殘酷的事實：熱門基線經常出乎意料地強，
許多看似複雜的模型其實贏不過它。若最終模型打不贏這條線，
那就該誠實寫出來並分析原因。

同時它也是冷啟動使用者的保底方案 —— 對一個沒有歷史的新使用者，
除了推熱門商品之外沒有更好的選擇。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from amazon_recsys.recall.base import take_top_k_unseen

SECONDS_PER_DAY = 86_400


@dataclass
class PopularityRecall:
    """依互動次數排序推薦。

    `window_days` 限定只統計 cutoff 之前這段期間的熱度。用全期熱度
    會讓十年前的暢銷品一直佔據榜首，對預測近期行為沒有幫助；
    90 天是常見的起點，之後可調。
    """

    window_days: int | None = 90
    pool_size: int = 2_000
    name: str = "popularity"
    _ranked: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def fit(self, con, src: str, cutoff: int) -> None:
        lo = cutoff - self.window_days * SECONDS_PER_DAY if self.window_days else None
        where = f"ts < {cutoff}" + (f" AND ts >= {lo}" if lo is not None else "")
        rows = con.execute(f"""
            SELECT item_idx
            FROM {src}
            WHERE {where}
            GROUP BY item_idx
            ORDER BY count(*) DESC, item_idx    -- item_idx 作為決勝鍵以確保可重現
            LIMIT {self.pool_size}
        """).fetchall()
        self._ranked = np.array([r[0] for r in rows], dtype=np.int64)

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        if self._ranked.size == 0:
            raise RuntimeError("尚未呼叫 fit()")
        return take_top_k_unseen(self._ranked, histories, k)
