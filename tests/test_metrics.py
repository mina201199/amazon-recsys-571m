"""評估指標測試。

指標是專案的尺。尺若有偏差，後面所有模型比較的結論都是錯的，
而且這種錯誤不會讓程式崩潰——它會安靜地給出好看的數字。

因此這裡的期望值全部是手算的，不是「跑一次看輸出是多少就寫進去」。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from amazon_recsys.evaluation import metrics as M

# 位置 1、2、3 的折扣：1/log2(pos+1)
D1 = 1.0 / math.log2(2)  # 1.00000
D2 = 1.0 / math.log2(3)  # 0.63093
D3 = 1.0 / math.log2(4)  # 0.50000


def test_perfect_ranking_scores_one():
    r = M.evaluate(np.array([[1, 2, 3]]), [{1, 2, 3}], ks=(3,))
    assert r.metrics["recall@3"] == pytest.approx(1.0)
    assert r.metrics["ndcg@3"] == pytest.approx(1.0)


def test_no_hits_scores_zero():
    r = M.evaluate(np.array([[4, 5, 6]]), [{1, 2, 3}], ks=(3,))
    assert r.metrics["recall@3"] == pytest.approx(0.0)
    assert r.metrics["ndcg@3"] == pytest.approx(0.0)


def test_ndcg_is_position_weighted():
    """同樣命中 1 個，排在前面必須得分更高——這是 NDCG 存在的意義。"""
    first = M.evaluate(np.array([[1, 8, 9]]), [{1}], ks=(3,)).metrics["ndcg@3"]
    third = M.evaluate(np.array([[8, 9, 1]]), [{1}], ks=(3,)).metrics["ndcg@3"]
    assert first == pytest.approx(D1 / D1)   # 命中在位置 1，IDCG = D1 → 1.0
    assert third == pytest.approx(D3 / D1)   # 命中在位置 3 → 0.5
    assert first > third


def test_ndcg_hand_computed_partial_hit():
    """recs=[9,1,9]、答案={1}：命中位置 2。"""
    r = M.evaluate(np.array([[9, 1, 9]]), [{1}], ks=(3,))
    assert r.metrics["ndcg@3"] == pytest.approx(D2 / D1)
    assert r.metrics["recall@3"] == pytest.approx(1.0)


def test_recall_divides_by_truth_size_not_k():
    """答案有 3 個但只推 2 個且都命中 → recall = 2/3，不是 1.0。"""
    r = M.evaluate(np.array([[1, 2]]), [{1, 2, 3}], ks=(2,))
    assert r.metrics["recall@2"] == pytest.approx(2 / 3)


def test_ndcg_idcg_capped_at_k():
    """清單只有 2 格、答案有 3 個時，前 2 格全中就該是滿分 1.0。

    若 IDCG 未以 K 為上限，這裡會算出小於 1 的分數，
    等於懲罰一個已經做到最好的模型。
    """
    r = M.evaluate(np.array([[1, 2]]), [{1, 2, 3}], ks=(2,))
    assert r.metrics["ndcg@2"] == pytest.approx(1.0)


def test_padding_never_counts_as_hit():
    """-1 是候選不足時的填充值，即使答案集含 -1 也不可命中。"""
    r = M.evaluate(np.array([[-1, -1, -1]]), [{-1, 1}], ks=(3,))
    assert r.metrics["recall@3"] == pytest.approx(0.0)


def test_users_without_truth_are_excluded_not_zeroed():
    """沒有正確答案的使用者應排除，不是當成 0 分——後者會低估模型。"""
    recs = np.array([[1, 2, 3], [4, 5, 6]])
    r = M.evaluate(recs, [{1, 2, 3}, set()], ks=(3,))
    assert r.n_users == 1
    assert r.metrics["recall@3"] == pytest.approx(1.0)


def test_coverage_detects_degenerate_popularity_model():
    """全部使用者都推同樣 3 個商品 → coverage 極低，即使 recall 完美。"""
    recs = np.tile([1, 2, 3], (100, 1))
    r = M.evaluate(recs, [{1, 2, 3}] * 100, ks=(3,), n_items=1000)
    assert r.metrics["recall@3"] == pytest.approx(1.0)
    assert r.metrics["coverage"] == pytest.approx(3 / 1000)


def test_averaging_is_over_users():
    """一人全中、一人全不中 → 平均 0.5。"""
    recs = np.array([[1, 2, 3], [7, 8, 9]])
    r = M.evaluate(recs, [{1, 2, 3}, {4, 5, 6}], ks=(3,))
    assert r.metrics["recall@3"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "recs, truths, err",
    [
        (np.array([1, 2, 3]), [{1}], "二維"),
        (np.array([[1, 2]]), [{1}, {2}], "不符"),
        (np.array([[1, 2]]), [{1}], "超過"),
    ],
)
def test_invalid_input_is_rejected(recs, truths, err):
    ks = (5,) if "超過" in err else (2,)
    with pytest.raises(ValueError, match=err):
        M.evaluate(recs, truths, ks=ks)


def test_paired_bootstrap_known_delta_and_reproducibility():
    baseline = np.array([[0], [0]])
    candidate = np.array([[1], [2]])
    truth = [{1}, {2}]
    result = M.paired_recall_bootstrap(baseline, candidate, truth, 1, samples=30)
    assert result["delta"] == result["ci95_low"] == result["ci95_high"] == 1.0
    assert result == M.paired_recall_bootstrap(baseline, candidate, truth, 1, samples=30)
    zero = M.paired_recall_bootstrap(baseline, baseline, truth, 1, samples=30)
    assert zero["ci95_low"] == zero["ci95_high"] == 0
