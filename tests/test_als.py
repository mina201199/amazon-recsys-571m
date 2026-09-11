"""ALS 召回測試。

合成資料構造成兩個興趣完全不重疊的族群。ALS 若真的學到結構，
就該只在族群內推薦；若沒學到，推薦會混雜兩群 —— 而那不會報錯。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall.als import ALSRecall
from amazon_recsys.recall.base import PAD

CUTOFF = S.ts("2023-06-01")
RECENT = S.ts("2023-05-01")

GROUP_A = [1, 2, 3, 4, 5, 6]        # 族群 A 的興趣
GROUP_B = [10, 11, 12, 13, 14, 15]  # 族群 B 的興趣


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE inter (user_idx INT, item_idx INT, ts INT)")
    rows = []
    for u in range(20):                       # 使用者 0-19 屬於族群 A
        rows += [(u, i, RECENT) for i in GROUP_A]
    for u in range(20, 40):                   # 使用者 20-39 屬於族群 B
        rows += [(u, i, RECENT) for i in GROUP_B]
    c.executemany("INSERT INTO inter VALUES (?, ?, ?)", rows)
    return c


def test_learns_group_structure(con):
    """買過族群 A 商品的人，推薦應落在族群 A 之內。"""
    r = ALSRecall(factors=16, iterations=20, min_user_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[1, 2, 3]], k=3)[0].tolist()
    assert set(out) <= set(GROUP_A), f"推薦混入了族群 B：{out}"
    assert not set(out) & {1, 2, 3}, "不應推薦已互動過的商品"


def test_symmetric_for_the_other_group(con):
    r = ALSRecall(factors=16, iterations=20, min_user_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[10, 11, 12]], k=3)[0].tolist()
    assert set(out) <= set(GROUP_B), f"推薦混入了族群 A：{out}"


def test_fold_in_works_for_unseen_user(con):
    """從未出現在訓練資料裡的使用者，仍應能透過商品因子取得推薦。"""
    r = ALSRecall(factors=16, iterations=20, min_user_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[4, 5]], k=2)[0].tolist()   # 這組歷史不屬於任何訓練使用者
    assert set(out) <= set(GROUP_A)
    assert PAD not in out


def test_unknown_items_yield_padding(con):
    """歷史商品全都不在模型中時，應回傳填充值而非崩潰——
    這類使用者交由熱門商品那一路處理。"""
    r = ALSRecall(factors=16, iterations=20, min_user_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert r.recommend([[9999]], k=3)[0].tolist() == [PAD] * 3
    assert r.recommend([[]], k=3)[0].tolist() == [PAD] * 3


def test_min_item_interactions_filters_rare_items(con):
    con.execute(f"INSERT INTO inter VALUES (99, 777, {RECENT})")   # 只出現一次
    r = ALSRecall(factors=16, iterations=10, min_item_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert 777 not in r._item_pos


def test_fit_ignores_data_after_cutoff(con):
    future = S.ts("2023-08-01")
    for u in range(50, 60):
        con.execute(
            f"INSERT INTO inter VALUES ({u}, 888, {future}), ({u}, 889, {future}),"
            f" ({u}, 890, {future}), ({u}, 891, {future}), ({u}, 892, {future})"
        )
    r = ALSRecall(factors=16, iterations=10)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert 888 not in r._item_pos, "統計到了 cutoff 之後的資料"


def test_raises_when_filters_remove_everything(con):
    r = ALSRecall(min_user_interactions=1000)
    with pytest.raises(ValueError, match="過濾後沒有互動"):
        r.fit(con, "inter", cutoff=CUTOFF)


def test_stats_reports_trained_size(con):
    r = ALSRecall(factors=16, iterations=10, min_user_interactions=5)
    r.fit(con, "inter", cutoff=CUTOFF)
    st = r.stats()
    assert st["users_trained"] == 40
    assert st["items_trained"] == 12
    assert st["factors"] == 16


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        ALSRecall().recommend([[1]], k=3)
