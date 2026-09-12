"""Implicit ALS 召回：標準 fold-in 與使用者／商品雙向分塊精確 Top-K。

以評論互動作隱式訊號，不把未互動視為明確負評。低互動門檻是成本與
覆蓋率的取捨，是否提升品質需實驗驗證。線上低延遲仍需 ANN 或預計算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from amazon_recsys.recall.base import PAD

SECONDS_PER_DAY = 86_400


@dataclass
class ALSRecall:
    """隱式回饋的交替最小平方法召回。"""

    factors: int = 64
    iterations: int = 15
    regularization: float = 0.05
    alpha: float = 40.0            # 隱式回饋的信心權重
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    window_days: int | None = None
    random_state: int = 42
    user_batch_size: int = 64
    item_block_size: int = 65_536
    max_score_mb: float = 128.0  # 分數及 Top-K 暫存預算，不是整個模型的記憶體上限
    name: str = "als"

    _model: object | None = None
    _item_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    _item_pos: dict[int, int] = field(default_factory=dict)

    def fit(self, con, src: str, cutoff: int) -> None:
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        lo = cutoff - self.window_days * SECONDS_PER_DAY if self.window_days else None
        where = f"ts < {cutoff}" + (f" AND ts >= {lo}" if lo is not None else "")

        # 先算出通過門檻的使用者與商品，再取互動 ——
        # 順序很重要：先篩選再建矩陣，才不會把整個矩陣建起來又丟掉。
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE als_input AS
            WITH base AS (SELECT user_idx, item_idx FROM {src} WHERE {where}),
                 ok_items AS (
                     SELECT item_idx FROM base
                     GROUP BY 1 HAVING count(*) >= {self.min_item_interactions}
                 ),
                 filtered AS (SELECT b.* FROM base b JOIN ok_items USING (item_idx)),
                 ok_users AS (
                     SELECT user_idx FROM filtered
                     GROUP BY 1 HAVING count(*) >= {self.min_user_interactions}
                 )
            SELECT f.user_idx, f.item_idx
            FROM filtered f JOIN ok_users USING (user_idx)
        """)

        rows = con.execute(
            "SELECT user_idx, item_idx FROM als_input"
        ).fetchnumpy()
        users_raw = rows["user_idx"].astype(np.int64)
        items_raw = rows["item_idx"].astype(np.int64)
        if users_raw.size == 0:
            raise ValueError(
                f"過濾後沒有互動（門檻：使用者 >= {self.min_user_interactions} 筆、"
                f"商品 >= {self.min_item_interactions} 筆）"
            )

        # 壓縮成連續索引：原始 ID 有數千萬的空洞，直接當矩陣索引會
        # 配置出遠大於必要的矩陣。
        u_uniq, u_pos = np.unique(users_raw, return_inverse=True)
        i_uniq, i_pos = np.unique(items_raw, return_inverse=True)
        self._item_index = i_uniq
        self._item_pos = {int(v): p for p, v in enumerate(i_uniq)}

        matrix = sp.csr_matrix(
            (np.ones(u_pos.size, dtype=np.float32), (u_pos, i_pos)),
            shape=(u_uniq.size, i_uniq.size),
        )

        # implicit 自己已做多執行緒平行；若底層 BLAS 再各自開 20 條，
        # 執行緒會互相搶佔，在全量資料上是數十分鐘與數小時的差別。
        # 建構與訓練都要包進來——implicit 的 BLAS 檢查是在 __init__ 執行的。
        with threadpool_limits(limits=1, user_api="blas"):
            self._model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=self.alpha,
                iterations=self.iterations,
                random_state=self.random_state,
                use_gpu=False,
            )
            self._model.fit(matrix, show_progress=False)

    def stats(self) -> dict[str, int]:
        if self._model is None:
            raise RuntimeError("尚未呼叫 fit()")
        return {
            "users_trained": int(self._model.user_factors.shape[0]),
            "items_trained": int(self._item_index.size),
            "factors": self.factors,
        }

    def recommend(self, histories: list[list[int]], k: int) -> np.ndarray:
        """解固定商品因子的 ALS 正規方程，再分塊搜尋精確 Top-K。

        implicit.recalculate_user 會使用訓練相同的 alpha 與 regularization。
        不再以商品向量平均冒充 fold-in。未知歷史回傳 PAD，留給其他通道。
        """
        from threadpoolctl import threadpool_limits

        if self._model is None:
            raise RuntimeError("尚未呼叫 fit()")
        if k < 1:
            raise ValueError("k 必須至少為 1")
        if self.user_batch_size < 1 or self.item_block_size < 1:
            raise ValueError("batch/block size 必須至少為 1")
        if not np.isfinite(self.max_score_mb) or self.max_score_mb <= 0:
            raise ValueError("max_score_mb 必須為正有限值")
        factors = np.asarray(self._model.item_factors)
        n_items = len(factors)
        out = np.full((len(histories), k), PAD, dtype=np.int64)
        # 保留 4 倍空間供分數、選取索引及暫存；模型因子與輸出另計。
        budget = max(1, int(self.max_score_mb * 1024**2) // (4 * factors.itemsize))
        user_batch = min(self.user_batch_size, budget)
        item_block = min(self.item_block_size, max(1, budget // user_batch))
        for start in range(0, len(histories), user_batch):
            hists = histories[start:start + user_batch]
            seen = [sorted({self._item_pos[i] for i in h if i in self._item_pos})
                    for h in hists]
            indices = np.array([i for row in seen for i in row], dtype=np.int32)
            indptr = np.concatenate(([0], np.cumsum([len(row) for row in seen])))
            interactions = sp.csr_matrix(
                (np.ones(len(indices), dtype=np.float32), indices, indptr),
                shape=(len(hists), n_items),
            )
            with threadpool_limits(limits=1, user_api="blas"):
                vectors = self._model.recalculate_user(np.arange(len(hists)), interactions)
            best_scores = [np.empty(0, dtype=factors.dtype) for _ in hists]
            best_items = [np.empty(0, dtype=np.int64) for _ in hists]
            for left in range(0, n_items, item_block):
                right = min(left + item_block, n_items)
                scores = vectors @ factors[left:right].T
                for row, known in enumerate(seen):
                    if not known:
                        continue
                    excluded = [i - left for i in known if left <= i < right]
                    scores[row, excluded] = -np.inf
                    block_ids = self._item_index[left:right]
                    ids, values = _top_k(block_ids, scores[row], k)
                    best_items[row], best_scores[row] = _top_k(
                        np.concatenate((best_items[row], ids)),
                        np.concatenate((best_scores[row], values)), k,
                    )
            for row, ids in enumerate(best_items):
                out[start + row, :len(ids)] = ids
        return out


def _top_k(ids: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """有限分數的精確 Top-K；同分依全域 item ID，包含切分邊界的同分。"""
    valid = np.flatnonzero(np.isfinite(scores))
    if valid.size > k:
        threshold = np.partition(scores[valid], valid.size - k)[valid.size - k]
        above = valid[scores[valid] > threshold]
        tied = valid[scores[valid] == threshold]
        tied = tied[np.argsort(ids[tied])[:k - len(above)]]
        valid = np.concatenate((above, tied))
    order = np.lexsort((ids[valid], -scores[valid]))
    chosen = valid[order]
    return ids[chosen], scores[chosen]
