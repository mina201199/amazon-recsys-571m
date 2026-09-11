"""時間切分與洩漏防護測試。

用合成資料驗證，因為真實資料無法構造出「剛好踩在邊界上」的情境，
而邊界正是這類錯誤發生的地方。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.evaluation import splits as S

SPLIT = S.TimeSplit(
    train_end=S.ts("2023-03-01"),
    valid_end=S.ts("2023-06-01"),
    test_end=S.ts("2023-09-10"),
)


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE inter (user_idx INT, item_idx INT, ts INT)")
    rows = [
        # 使用者 0：有歷史、測試段買了新商品 → 應被納入評估
        (0, 10, S.ts("2022-01-01")),   # 早期歷史
        (0, 12, S.ts("2023-04-01")),   # 驗證段 → 評估測試段時仍算歷史
        (0, 11, S.ts("2023-07-01")),   # 測試段、新商品 → 正確答案
        (0, 10, S.ts("2023-07-05")),   # 測試段、但已買過 → 應排除
        # 使用者 1：只有歷史，測試段沒動作 → 應排除
        (1, 20, S.ts("2022-05-01")),
        # 使用者 2：只有測試段互動，沒有歷史 → 應排除
        (2, 30, S.ts("2023-07-01")),
    ]
    c.executemany("INSERT INTO inter VALUES (?, ?, ?)", rows)
    return c


# --------------------------------------------------------------------------
# 邊界定義
# --------------------------------------------------------------------------
def test_boundaries_must_increase():
    with pytest.raises(ValueError, match="遞增"):
        S.TimeSplit(train_end=300, valid_end=200, test_end=100)


def test_feature_cutoff_is_segment_start():
    """評估某段時，特徵只能用到該段起點——這是防洩漏的核心規則。"""
    assert SPLIT.feature_cutoff("valid") == SPLIT.train_end
    assert SPLIT.feature_cutoff("test") == SPLIT.valid_end


def test_train_segment_cannot_be_evaluated(con):
    with pytest.raises(ValueError, match="訓練段"):
        S.build_eval_set(con, "inter", SPLIT, segment="train")


# --------------------------------------------------------------------------
# 評估集建構
# --------------------------------------------------------------------------
def test_eval_set_selects_only_qualifying_users(con):
    users, _, truths = S.build_eval_set(con, "inter", SPLIT, segment="test")
    assert users == [0], "只有使用者 0 同時具備歷史與測試段的新商品"
    assert truths == [{11}]


def test_history_includes_everything_before_cutoff(con):
    """測試段的 cutoff 是 valid_end，所以驗證段的互動仍屬於歷史。

    這點很容易寫錯成「只取訓練段」，那會白白丟掉三個月的訊號。
    """
    _, hists, _ = S.build_eval_set(con, "inter", SPLIT, segment="test")
    assert sorted(hists[0]) == [10, 12]


def test_history_never_contains_future_data(con):
    """結構性保證：歷史中的任何互動都必須早於 cutoff。"""
    cutoff = SPLIT.feature_cutoff("test")
    users, hists, _ = S.build_eval_set(con, "inter", SPLIT, segment="test")
    for uid, hist in zip(users, hists, strict=True):
        for item in hist:
            earliest = con.execute(
                "SELECT min(ts) FROM inter WHERE user_idx = ? AND item_idx = ?",
                [uid, item],
            ).fetchone()[0]
            assert earliest < cutoff, f"歷史商品 {item} 沒有任何早於 cutoff 的互動"


def test_truth_excludes_previously_seen_items(con):
    """使用者 0 在測試段重買了商品 10，但他早就買過 → 不算正確答案。

    留著它會讓分數虛高：推薦一個對方早就買過的東西不算本事。
    """
    _, _, truths = S.build_eval_set(con, "inter", SPLIT, segment="test")
    assert 10 not in truths[0]
    assert truths[0] == {11}


def test_min_history_filter(con):
    users, _, _ = S.build_eval_set(con, "inter", SPLIT, segment="test", min_history=3)
    assert users == [], "使用者 0 只有 2 筆歷史，門檻設 3 應被濾掉"


# --------------------------------------------------------------------------
# 洩漏守門員
# --------------------------------------------------------------------------
def test_assert_no_leakage_passes_on_clean_table(con):
    con.execute(f"CREATE TABLE clean AS SELECT * FROM inter WHERE ts < {SPLIT.train_end}")
    S.assert_no_leakage(con, "clean", SPLIT.train_end)  # 不應拋錯


def test_assert_no_leakage_catches_future_rows(con):
    with pytest.raises(AssertionError, match="資料洩漏"):
        S.assert_no_leakage(con, "inter", SPLIT.train_end)


def test_split_summary_counts_each_segment(con):
    rows = {r["segment"]: r for r in S.split_summary(con, "inter", SPLIT)}
    assert rows["train"]["interactions"] == 2   # (0,10) 與 (1,20)
    assert rows["valid"]["interactions"] == 1   # (0,12)
    assert rows["test"]["interactions"] == 3    # (0,11) (0,10) (2,30)
