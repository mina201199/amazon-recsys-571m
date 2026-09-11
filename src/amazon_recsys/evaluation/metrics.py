"""推薦系統評估指標。

這是整個專案的「尺」。先把尺做對，後面每個模型才有意義的比較基準。

指標選擇的理由（見 docs/specs/ §8）：

- **Recall@K**：推薦的 K 個裡命中多少比例的正確答案。直觀、好解釋。
- **NDCG@K**：命中的位置也算分——排第 1 名比排第 10 名有價值。
- **Catalogue coverage**：模型總共推薦過多少種不同商品。
  只會推 100 種商品的系統，Recall 再高也是廢的；這個指標會抓出
  「全部推熱門商品」的退化行為，而前兩個指標抓不到。

全部以 numpy 向量化實作，避免逐使用者的 Python 迴圈。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    """一次評估的完整結果。"""

    n_users: int
    metrics: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"{k}={v:.4f}" for k, v in sorted(self.metrics.items())]
        return f"[{self.n_users:,} 位使用者] " + "  ".join(parts)


def _hit_matrix(recs: np.ndarray, truths: list[set[int]]) -> np.ndarray:
    """回傳 (n_users, k) 的布林矩陣，標示每個推薦位置是否命中。

    recs 中的 -1 代表「沒有推薦」（候選不足時的填充值），永不算命中。
    """
    hits = np.zeros(recs.shape, dtype=bool)
    for u, truth in enumerate(truths):
        if not truth:
            continue
        row = recs[u]
        # np.isin 對小集合有額外開銷，這裡用 set 查找反而更快
        hits[u] = [(item in truth) and item >= 0 for item in row]
    return hits


def recall_at_k(hits: np.ndarray, truth_sizes: np.ndarray, k: int) -> float:
    """Recall@K：命中數 / 正確答案總數，對使用者取平均。

    只計入有正確答案的使用者——把沒有答案的使用者算成 0 分會低估模型。
    """
    valid = truth_sizes > 0
    if not valid.any():
        return 0.0
    n_hit = hits[:, :k].sum(axis=1)
    return float(np.mean(n_hit[valid] / truth_sizes[valid]))


def ndcg_at_k(hits: np.ndarray, truth_sizes: np.ndarray, k: int) -> float:
    """NDCG@K（二元相關性）。

    DCG  = Σ 1/log2(位置+1)，位置從 1 起算
    IDCG = 理想排序下的 DCG，即前 min(答案數, K) 個位置全部命中
    """
    valid = truth_sizes > 0
    if not valid.any():
        return 0.0

    discounts = 1.0 / np.log2(np.arange(2, k + 2))          # 位置 1..k 的折扣
    dcg = (hits[:, :k] * discounts).sum(axis=1)

    # IDCG：前 min(n_truth, k) 個位置的折扣累加
    cum = np.concatenate([[0.0], np.cumsum(discounts)])      # cum[i] = 前 i 個的和
    idcg = cum[np.minimum(truth_sizes, k)]

    return float(np.mean(dcg[valid] / idcg[valid]))


def catalogue_coverage(recs: np.ndarray, n_items: int) -> float:
    """推薦清單涵蓋了商品目錄的多少比例。

    這個指標存在的理由：一個只推 100 種熱門商品的系統，
    Recall 可能不難看，但商業上毫無價值。Recall 和 NDCG 都抓不到這件事。
    """
    if n_items <= 0:
        return 0.0
    return float(len(np.unique(recs[recs >= 0])) / n_items)


def evaluate(
    recs: np.ndarray,
    truths: list[set[int]],
    ks: tuple[int, ...] = (10,),
    n_items: int | None = None,
) -> EvalResult:
    """計算全部指標。

    參數
    ----
    recs
        形狀 (n_users, k_max) 的推薦矩陣，每列是該使用者的排序推薦清單。
        用 -1 填充不足的位置。
    truths
        每位使用者的正確答案集合，順序需與 recs 的列對應。
    ks
        要計算的 K 值。
    n_items
        商品目錄總數，提供時才計算 coverage。
    """
    recs = np.asarray(recs)
    if recs.ndim != 2:
        raise ValueError(f"recs 應為二維陣列，收到 shape={recs.shape}")
    if len(truths) != recs.shape[0]:
        raise ValueError(f"truths 長度 {len(truths)} 與 recs 列數 {recs.shape[0]} 不符")
    if max(ks) > recs.shape[1]:
        raise ValueError(f"K={max(ks)} 超過推薦清單長度 {recs.shape[1]}")

    hits = _hit_matrix(recs, truths)
    truth_sizes = np.array([len(t) for t in truths])

    result = EvalResult(n_users=int((truth_sizes > 0).sum()))
    for k in ks:
        result.metrics[f"recall@{k}"] = recall_at_k(hits, truth_sizes, k)
        result.metrics[f"ndcg@{k}"] = ndcg_at_k(hits, truth_sizes, k)
    if n_items:
        result.metrics["coverage"] = catalogue_coverage(recs, n_items)
    return result
