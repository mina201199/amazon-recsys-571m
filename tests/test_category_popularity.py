"""類別熱門召回測試。

全域熱門在驗證集上只用了約 507 件商品就拿到 Recall@500 = 0.0253，是最強的
單路通道。它的弱點很明確：不管使用者買什麼，推薦的都是同一份清單。

這一路只改一件事——把熱度限制在使用者買過的類別裡。所以測試的核心是
**全域最熱的商品不該出現在只買過別的類別的使用者候選中**；
若它照樣擠進來，這個通道就退化成全域熱門，而且不會有任何錯誤訊息。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall.base import PAD
from amazon_recsys.recall.category_popularity import CategoryPopularityRecall

CUTOFF = S.ts("2023-06-01")
RECENT = S.ts("2023-05-01")
AFTER = S.ts("2023-08-01")

CAT_BOOKS, CAT_TOOLS = 1, 2
# 類別 1：商品 1-4，熱度 12 / 8 / 4 / 2
# 類別 2：商品 99 熱度 40（全域最熱）、商品 98 熱度 30
BOOK_POPS = {1: 12, 2: 8, 3: 4, 4: 2}
TOOL_POPS = {99: 40, 98: 30}
TOOL_ITEM = 99


@pytest.fixture
def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE inter (user_idx INT, item_idx INT, category_idx INT, ts INT)")
    rows = []
    for cat, pops in ((CAT_BOOKS, BOOK_POPS), (CAT_TOOLS, TOOL_POPS)):
        for item, pop in pops.items():
            rows += [(1000 + len(rows) + i, item, cat, RECENT) for i in range(pop)]
    c.executemany("INSERT INTO inter VALUES (?, ?, ?, ?)", rows)
    return c


def _fit(con, **kw):
    r = CategoryPopularityRecall(**kw)
    r.fit(con, "inter", cutoff=CUTOFF)
    return r


def test_global_top_item_is_not_recommended_outside_its_category(con):
    """只買過書的人，不該拿到全域最熱的工具——這是本通道存在的全部理由。"""
    out = _fit(con).recommend([[1]], k=3)
    assert TOOL_ITEM not in out[0].tolist()
    assert out[0].tolist() == [2, 3, 4]        # 同類別次熱門，且排除已買的 1


def test_falls_back_to_padding_when_category_is_unknown(con):
    """歷史商品不在任何已知類別時回傳 PAD，交給其他通道處理。"""
    out = _fit(con).recommend([[55_555]], k=3)
    assert out[0].tolist() == [PAD] * 3


def test_mixes_categories_by_how_much_the_user_bought_in_each(con):
    """買 3 本書、1 個工具的人：剩下的書應排在剩下的工具前面。

    書 4 在類別內名次 4、工具 98 名次 2，若只看類別內名次，工具會贏；
    乘上使用者在該類別的購買次數（3 比 1）之後，書才排到前面。
    """
    out = _fit(con).recommend([[1, 2, 3, TOOL_ITEM]], k=2)
    assert out[0].tolist() == [4, 98]


def test_popularity_ignores_interactions_after_cutoff(con):
    """cutoff 之後的互動不得計入熱度，否則等於偷看未來。"""
    con.executemany("INSERT INTO inter VALUES (?, ?, ?, ?)",
                    [(9_000 + i, 3, CAT_BOOKS, AFTER) for i in range(100)])
    out = _fit(con).recommend([[1]], k=2)
    assert out[0].tolist() == [2, 3]           # 3 仍排在 2 之後


def test_excludes_items_the_user_already_interacted_with(con):
    out = _fit(con).recommend([[1, 2]], k=3)
    assert 1 not in out[0].tolist() and 2 not in out[0].tolist()


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        CategoryPopularityRecall().recommend([[1]], k=3)
