"""共現召回測試。

合成資料刻意構造成「原始次數會打平、正規化後才分得出高下」的情境，
因為那正是這個模組最容易悄悄失效的地方。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall.base import PAD
from amazon_recsys.recall.covisitation import CoVisitationRecall

CUTOFF = S.ts("2023-06-01")
RECENT = S.ts("2023-05-01")
OLD = S.ts("2019-01-01")


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE inter (user_idx INT, item_idx INT, ts INT)")
    rows = []
    # 使用者 0-9 同時買了商品 1、2 和 99
    for u in range(10):
        rows += [(u, 1, RECENT), (u, 2, RECENT), (u, 99, RECENT)]
    # 使用者 10-49 只買了商品 99 → 99 變成熱門商品（pop=50）
    for u in range(10, 50):
        rows.append((u, 99, RECENT))
    c.executemany("INSERT INTO inter VALUES (?, ?, ?)", rows)
    return c


def test_normalisation_beats_raw_popularity(con):
    """商品 1 與 2、1 與 99 的共現次數都是 10，原始次數無法區分。

    正規化後：
        score(1->2)  = 10 / sqrt(10 * 10) = 1.000
        score(1->99) = 10 / sqrt(10 * 50) = 0.447
    所以買過商品 1 的人，應該先被推薦 2 而不是熱門的 99。
    """
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[1]], k=2)
    assert out[0, 0] == 2, "正規化失效——熱門商品搶走了第一名"
    assert out[0, 1] == 99


def test_excludes_items_already_in_history(con):
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[1, 2]], k=3)
    assert 1 not in out[0].tolist()
    assert 2 not in out[0].tolist()
    assert out[0, 0] == 99


def test_min_cooccurrence_filters_coincidences(con):
    """只共現一次的配對是巧合，不是訊號。"""
    con.execute(f"INSERT INTO inter VALUES (999, 1, {RECENT}), (999, 777, {RECENT})")
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert 777 not in r.recommend([[1]], k=10)[0].tolist()

    r_loose = CoVisitationRecall(min_cooccurrence=1)
    r_loose.fit(con, "inter", cutoff=CUTOFF)
    assert 777 in r_loose.recommend([[1]], k=10)[0].tolist()


def test_fit_ignores_data_after_cutoff(con):
    """cutoff 之後的互動不得進入共現統計——這是防洩漏的關鍵。"""
    future = S.ts("2023-08-01")
    con.execute(f"INSERT INTO inter VALUES (500, 1, {future}), (500, 888, {future})")
    con.execute(f"INSERT INTO inter VALUES (501, 1, {future}), (501, 888, {future})")
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert 888 not in r.recommend([[1]], k=10)[0].tolist(), "統計到了未來資料"


def test_window_excludes_stale_cooccurrence(con):
    """視窗外的舊共現不應被統計。"""
    for u in range(600, 610):
        con.execute(f"INSERT INTO inter VALUES ({u}, 1, {OLD}), ({u}, 555, {OLD})")
    r = CoVisitationRecall(window_days=365, min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    assert 555 not in r.recommend([[1]], k=10)[0].tolist()

    r_all = CoVisitationRecall(window_days=None, min_cooccurrence=2)
    r_all.fit(con, "inter", cutoff=CUTOFF)
    assert 555 in r_all.recommend([[1]], k=10)[0].tolist()


def test_max_items_per_user_keeps_most_recent(con):
    """限制每人取樣數時，保留的必須是最近的互動，不是任意的。"""
    con.execute("DELETE FROM inter")
    # 使用者 0 依序買了 10(最舊) .. 14(最新)，另有使用者與其中部分共現
    for i, item in enumerate([10, 11, 12, 13, 14]):
        con.execute(f"INSERT INTO inter VALUES (0, {item}, {RECENT - (5 - i) * 86400})")
    for u in range(1, 5):
        con.execute(f"INSERT INTO inter VALUES ({u}, 14, {RECENT}), ({u}, 13, {RECENT})")

    r = CoVisitationRecall(max_items_per_user=2, min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    # 使用者 0 只保留最近兩筆 (13, 14)；商品 10 不該有任何鄰居
    assert r.recommend([[10]], k=5)[0].tolist() == [PAD] * 5


def test_empty_history_returns_padding(con):
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    out = r.recommend([[]], k=3)
    assert out.tolist() == [[PAD, PAD, PAD]]


def test_stats_reports_graph_size(con):
    r = CoVisitationRecall(min_cooccurrence=2)
    r.fit(con, "inter", cutoff=CUTOFF)
    st = r.stats()
    assert st["items_with_neighbours"] == 3      # 商品 1、2、99
    assert st["edges"] == 6                      # 三個商品兩兩對稱 = 3 * 2


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        CoVisitationRecall().recommend([[1]], k=3)
