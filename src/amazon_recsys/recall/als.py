"""ALS 矩陣分解召回。

共現召回只看得到「直接一起被買過」的商品。ALS 把使用者與商品投影到
同一個低維空間，能找出沒有直接共現、但興趣結構相近的商品 ——
兩路互補，這是多路召回的意義。

## 必須先算的成本帳

    互動矩陣      5451 萬使用者 × 4819 萬商品，5.71 億個非零元素
    稀疏矩陣      約 4.8 GB                      -> 可行
    潛在因子      64 維時使用者 14 GB + 商品 12 GB -> 26 GB，吃緊但可行
    每輪迭代      5451 萬 × 64^3 的 Cholesky 求解 -> 約 12 分鐘/輪

15 輪迭代要 3 小時，在只有幾週的專案裡不合理。

## 解法：過濾低互動使用者

`min_user_interactions` 預設為 5。這不是為了省時間而犧牲品質 ——
兩者方向一致：只買過一兩樣東西的使用者，他的潛在向量本來就估不準，
留在訓練集裡既拖慢求解又貢獻雜訊。

被濾掉的使用者不會沒有推薦：評估時採 fold-in（用已訓練好的商品因子
反推該使用者的向量），沒有歷史的則由熱門商品那一路接手。
這正是多路召回的分工。
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
        """以 fold-in 為每位使用者計算推薦。

        不查訓練時的使用者因子，而是用其歷史商品的因子即時反推向量。
        這樣被門檻濾掉的使用者一樣有推薦，也保證推薦只依賴
        cutoff 之前的歷史。
        """
        if self._model is None:
            raise RuntimeError("尚未呼叫 fit()")

        item_factors = np.asarray(self._model.item_factors)   # (n_items, factors)
        out = np.full((len(histories), k), PAD, dtype=np.int64)
        n_items = item_factors.shape[0]

        # 先把所有使用者向量疊成一個矩陣，再一次做矩陣相乘。
        #
        # 原本的寫法是逐一使用者計算 item_factors @ vec。兩萬位使用者
        # 乘上 1152 萬個商品、32 個因子，是 7.8 兆次運算，且每次都要
        # 重新走訪整個因子矩陣——實測 recommend 花了 53 分鐘。
        # 疊成矩陣後交給 BLAS 一次算完，記憶體與快取的利用率完全不同。
        user_vecs = np.zeros((len(histories), item_factors.shape[1]), dtype=item_factors.dtype)
        seen: list[list[int]] = []
        for u, hist in enumerate(histories):
            pos = [self._item_pos[i] for i in hist if i in self._item_pos]
            seen.append(pos)
            if pos:
                # 使用者向量 = 其歷史商品因子的平均（fold-in 的簡化形式）
                user_vecs[u] = item_factors[pos].mean(axis=0)

        # 分批處理，避免一次配置 (n_users, n_items) 的分數矩陣
        batch = max(1, min(len(histories), 8_000_000 // max(n_items, 1) or 1))
        for start in range(0, len(histories), batch):
            stop = min(start + batch, len(histories))
            scores = user_vecs[start:stop] @ item_factors.T      # (batch, n_items)
            for row in range(stop - start):
                u = start + row
                if not seen[u]:
                    continue
                s = scores[row]
                s[seen[u]] = -np.inf                 # 排除已互動過的商品
                top = np.argpartition(-s, min(k, s.size - 1))[:k]
                top = top[np.argsort(-s[top])]
                valid = top[np.isfinite(s[top])]
                out[u, : valid.size] = self._item_index[valid]
        return out
