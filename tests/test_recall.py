"""召回層測試。"""

from __future__ import annotations

import duckdb
import numpy as np
import pytest

from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall import base
from amazon_recsys.recall.popularity import PopularityRecall


# --------------------------------------------------------------------------
# take_top_k_unseen
# --------------------------------------------------------------------------
def test_excludes_items_already_in_history():
    pool = np.array([1, 2, 3, 4, 5])
    out = base.take_top_k_unseen(pool, [[2, 4]], k=3)
    assert out.tolist() == [[1, 3, 5]]


def test_pads_when_pool_is_exhausted():
    pool = np.array([1, 2, 3])
    out = base.take_top_k_unseen(pool, [[1, 2]], k=3)
    assert out.tolist() == [[3, base.PAD, base.PAD]]


def test_users_are_independent():
    pool = np.array([1, 2, 3])
    out = base.take_top_k_unseen(pool, [[1], [3]], k=2)
    assert out.tolist() == [[2, 3], [1, 2]]


# --------------------------------------------------------------------------
# merge_channels
# --------------------------------------------------------------------------
def test_merge_interleaves_channels_round_robin():
    """先各取第 1 名、再各取第 2 名，避免某一路獨占名額。"""
    a = np.array([[10, 11, 12]])
    b = np.array([[20, 21, 22]])
    out = base.merge_channels([a, b], k=4)
    assert out.tolist() == [[10, 20, 11, 21]]


def test_merge_deduplicates_across_channels():
    """重複的候選被跳過，該通道不會因此獲得補位機會。

    深度 0：a 取 10；b 也是 10，已存在 → 跳過。
    深度 1：a 取 11；b 取 12。
    結果為 [10, 11, 12] —— 而非 [10, 12, 11]，
    後者等於讓 b 補回被跳過的名額，那不是輪流交錯的語意。
    """
    a = np.array([[10, 11]])
    b = np.array([[10, 12]])   # 首位重複
    out = base.merge_channels([a, b], k=3)
    assert out.tolist() == [[10, 11, 12]]


def test_merge_ignores_padding():
    a = np.array([[10, base.PAD]])
    b = np.array([[20, 21]])
    out = base.merge_channels([a, b], k=3)
    assert out.tolist() == [[10, 20, 21]]


def test_merge_pads_when_candidates_run_out():
    a = np.array([[10]])
    b = np.array([[20]])
    out = base.merge_channels([a, b], k=4)
    assert out.tolist() == [[10, 20, base.PAD, base.PAD]]


def test_merge_rejects_mismatched_user_counts():
    with pytest.raises(ValueError, match="使用者數"):
        base.merge_channels([np.zeros((2, 3)), np.zeros((3, 3))], k=2)


# --------------------------------------------------------------------------
# PopularityRecall
# --------------------------------------------------------------------------
@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE inter (user_idx INT, item_idx INT, ts INT)")
    rows = []
    recent = S.ts("2023-05-01")   # 在 90 天視窗內
    old = S.ts("2020-01-01")      # 視窗外
    # 商品 1 近期最熱（5 次），商品 2 次之（3 次）
    rows += [(u, 1, recent) for u in range(5)]
    rows += [(u, 2, recent) for u in range(3)]
    # 商品 99 在久遠以前有 100 次互動 —— 全期熱度會讓它奪冠
    rows += [(u, 99, old) for u in range(100)]
    c.executemany("INSERT INTO inter VALUES (?, ?, ?)", rows)
    return c


def test_recent_window_ignores_stale_bestsellers(con):
    """十年前的暢銷品不該佔據榜首——這正是限定時間視窗的理由。"""
    r = PopularityRecall(window_days=90)
    r.fit(con, "inter", cutoff=S.ts("2023-06-01"))
    assert r._ranked[:2].tolist() == [1, 2]
    assert 99 not in r._ranked.tolist()


def test_full_history_window_includes_old_items(con):
    r = PopularityRecall(window_days=None)
    r.fit(con, "inter", cutoff=S.ts("2023-06-01"))
    assert r._ranked[0] == 99, "全期統計下，累積 100 次的商品應排第一"


def test_fit_respects_cutoff(con):
    """cutoff 之後的資料不可被統計進去。"""
    r = PopularityRecall(window_days=None)
    r.fit(con, "inter", cutoff=S.ts("2021-01-01"))
    assert r._ranked.tolist() == [99], "只有 2020 年的互動早於 cutoff"


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        PopularityRecall().recommend([[1]], k=3)


def test_recommendations_exclude_history(con):
    r = PopularityRecall(window_days=90)
    r.fit(con, "inter", cutoff=S.ts("2023-06-01"))
    out = r.recommend([[1]], k=1)
    assert out.tolist() == [[2]], "商品 1 已在歷史中，應推第二熱門的 2"


def test_rrf_rewards_agreement_and_respects_zero_weight():
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 2, 5]])
    assert base.weighted_rrf([a, b], 3)[0, 0] == 2
    assert base.weighted_rrf([a, b], 3, [1, 0]).tolist() == [[1, 2, 3]]


def test_rrf_duplicate_padding_and_ties():
    a = np.array([[1, 1, -1]])
    b = np.array([[2, -1, -1]])
    assert base.weighted_rrf([a, b], 4).tolist() == [[1, 2, -1, -1]]


@pytest.mark.parametrize("weights", [[0, 0], [-1, 1], [1], [float("nan"), 1]])
def test_rrf_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        base.weighted_rrf([np.array([[1]]), np.array([[2]])], 2, weights)
