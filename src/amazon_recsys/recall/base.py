"""召回層的共用介面。

召回層負責把 4819 萬商品縮到幾百個候選。它的評估指標是 Recall@500：
正確答案有沒有落在候選集裡。**這是整個系統的天花板** —— 召回沒撈到的
商品，排序層再強也救不回來。

所有召回通道實作同一個介面，方便組合與比較。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

PAD = -1  # 候選不足時的填充值；評估時永不算命中


@runtime_checkable
class RecallChannel(Protocol):
    """一路召回通道。"""

    name: str

    def fit(self, con, src: str, cutoff: int) -> None:
        """從 ts < cutoff 的資料建立所需的統計。

        cutoff 由 TimeSplit.feature_cutoff() 提供；實作**不得**讀取
        該時間點之後的任何資料。
        """
        ...

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        """為每位使用者產生 k 個候選，回傳形狀 (n_users, k) 的陣列。

        不足 k 個時以 PAD 填充。已在 history 中的商品必須排除。
        """
        ...


def take_top_k_unseen(
    ranked_pool: np.ndarray,
    histories: list[list[int]],
    k: int,
) -> np.ndarray:
    """從一份全域排序清單中，為每位使用者取前 k 個他沒看過的商品。

    `ranked_pool` 需比 k 長，以容納被濾掉的已看商品；若某位使用者的
    歷史耗盡了整個池子，剩餘位置以 PAD 填充。
    """
    out = np.full((len(histories), k), PAD, dtype=np.int64)
    for u, hist in enumerate(histories):
        seen = set(hist)
        row = [i for i in ranked_pool if i not in seen]
        n = min(k, len(row))
        out[u, :n] = row[:n]
    return out


def merge_channels(
    candidate_lists: list[np.ndarray],
    k: int,
) -> np.ndarray:
    """合併多路召回的結果，去重後保留各通道的相對順序。

    採輪流交錯（round-robin）：先各取第 1 名、再各取第 2 名……
    這樣不會讓某一路通道獨占候選名額。
    """
    if not candidate_lists:
        raise ValueError("至少需要一路召回結果")
    n_users = candidate_lists[0].shape[0]
    if any(c.shape[0] != n_users for c in candidate_lists):
        raise ValueError("各通道的使用者數必須一致")

    out = np.full((n_users, k), PAD, dtype=np.int64)
    depth = max(c.shape[1] for c in candidate_lists)

    for u in range(n_users):
        seen: set[int] = set()
        picked: list[int] = []
        for d in range(depth):
            for cand in candidate_lists:
                if d >= cand.shape[1] or len(picked) >= k:
                    continue
                item = int(cand[u, d])
                if item != PAD and item not in seen:
                    seen.add(item)
                    picked.append(item)
            if len(picked) >= k:
                break
        out[u, : len(picked)] = picked[:k]
    return out
