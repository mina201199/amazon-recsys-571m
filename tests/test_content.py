"""內容式召回測試。

這一路存在的唯一理由是碰得到其他通道碰不到的商品——切分點前零互動的
冷啟動商品。所以測試的核心不是「有沒有推薦」，而是**冷啟動商品是否
真的出現在候選中**：若它被熱度排序悄悄擠掉，這個通道就沒有意義，
而且不會有任何錯誤訊息。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall.base import PAD
from amazon_recsys.recall.content import ContentRecall

CUTOFF = S.ts("2023-06-01")
BEFORE = S.ts("2023-01-01")
AFTER = S.ts("2023-08-01")

# 品牌 1：商品 1-3 有互動、商品 100 完全沒有（冷啟動）
# 品牌 2：商品 10-11，與品牌 1 同類別但價位高得多
STORE_1 = [1, 2, 3]
COLD_IN_STORE_1 = 100
STORE_2 = [10, 11]


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("""
        CREATE TABLE items (
            item_idx BIGINT, main_category_idx SMALLINT,
            store_idx INTEGER, price DOUBLE
        )
    """)
    rows = [(i, 1, 1, 20.0) for i in STORE_1]
    rows.append((COLD_IN_STORE_1, 1, 1, 22.0))      # 同品牌、同價位，但零互動
    rows += [(i, 1, 2, 500.0) for i in STORE_2]     # 同類別但價位差 25 倍
    c.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", rows)

    c.execute("CREATE TABLE inter (user_idx INT, item_idx BIGINT, ts INT)")
    inter = []
    for u in range(50):                              # 讓 1-3 與 10-11 都有熱度
        inter += [(u, i, BEFORE) for i in [*STORE_1, *STORE_2]]
    # 冷啟動商品只在切分點之後出現，切分點前熱度為 0
    inter.append((999, COLD_IN_STORE_1, AFTER))
    c.executemany("INSERT INTO inter VALUES (?, ?, ?)", inter)
    return c


def _fit(con, **kw):
    r = ContentRecall(items_table="items", **kw)
    r.fit(con, "inter", cutoff=CUTOFF)
    return r


def test_cold_item_is_reachable(con):
    """核心測試：零互動的商品必須出現在候選中。

    其他三路都碰不到它——沒有熱度、沒有共現、沒有潛在向量。
    若這裡也撈不到，這個通道就沒有存在意義。
    """
    r = _fit(con)
    rec = r.recommend([[1]], k=10)[0].tolist()
    assert COLD_IN_STORE_1 in rec, f"冷啟動商品沒有進候選：{rec}"


def test_cold_slots_zero_drops_cold_item(con):
    """把保留名額設為 0，冷啟動商品就會被熱度擠掉——證明保留機制確實在起作用。

    類別池也必須一併收緊。否則在只有 6 件商品的合成資料上，
    per_category=400 會把冷啟動商品從另一條路徑放進來，
    測試就測不到品牌池的保留機制。
    """
    r = _fit(con, cold_slots=0, per_store=2, per_category=2)
    rec = r.recommend([[1]], k=10)[0].tolist()
    assert COLD_IN_STORE_1 not in rec


def test_same_brand_ranks_above_same_category(con):
    """同品牌是最強的內容訊號，應排在只有同類別的商品之前。"""
    r = _fit(con)
    rec = r.recommend([[1]], k=4)[0].tolist()
    same_store = [i for i in rec if i in [*STORE_1, COLD_IN_STORE_1]]
    other_store = [i for i in rec if i in STORE_2]
    if other_store:
        assert rec.index(same_store[0]) < rec.index(other_store[0])


def test_excludes_items_already_seen(con):
    r = _fit(con)
    rec = r.recommend([[1, 2]], k=10)[0].tolist()
    assert 1 not in rec and 2 not in rec


def test_price_band_favours_similar_price(con):
    """使用者只買過 20 元的商品，500 元的同類別商品不該排在前面。"""
    r = _fit(con)
    rec = r.recommend([[1]], k=2)[0].tolist()
    assert not set(rec) & set(STORE_2), f"價位相差 25 倍的商品排進前兩名：{rec}"


def test_popularity_ignores_post_cutoff_interactions(con):
    """熱度必須只由切分點前的互動計算。

    冷啟動商品在切分點後有互動；若熱度算進去，它就不再被視為冷啟動，
    保留名額的機制也會失效。
    """
    r = _fit(con)
    pop = con.execute(
        f"SELECT pop FROM {r._items} WHERE item_idx = {COLD_IN_STORE_1}"
    ).fetchone()[0]
    assert pop == 0, "熱度統計到了切分點之後的互動"


def test_unknown_history_returns_padding(con):
    """歷史商品不在屬性表中時回傳填充值，交由其他通道處理。"""
    r = _fit(con)
    assert r.recommend([[77777]], k=3)[0].tolist() == [PAD] * 3
    assert r.recommend([[]], k=3)[0].tolist() == [PAD] * 3


def test_stats_reports_cold_count(con):
    r = _fit(con)
    st = r.stats()
    assert st["items_with_attributes"] == 6
    assert st["cold_items"] == 1


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        ContentRecall(items_table="items").recommend([[1]], k=3)
