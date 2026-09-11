"""互動表轉檔的測試。

這層是整條管線的地基：若 ID 映射不穩定或時間戳被寫錯，
錯誤會一路傳播到召回與排序，且很難回溯。因此這裡驗證的是
**不變量**，而不只是「跑得完」。

測試用已下載的最小類別，沒有檔案時自動跳過。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys import config
from amazon_recsys.ingest import build_interactions as bi

TEST_CATS = ("Subscription_Boxes", "Magazine_Subscriptions")

pytestmark = pytest.mark.skipif(
    not all((config.RAW_REVIEWS_DIR / f"{c}.jsonl.gz").exists() for c in TEST_CATS),
    reason="測試用的原始檔尚未下載",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("interactions")
    con = bi.connect(memory_limit="4GB")
    stats = bi.build(con, categories=TEST_CATS, out_dir=out)
    return out, stats


def _read(out):
    con = duckdb.connect()
    return con, f"read_parquet('{out.as_posix()}/**/*.parquet', hive_partitioning=true)"


def test_dedup_removes_duplicate_user_item_pairs(built):
    out, stats = built
    assert stats.kept_rows <= stats.raw_rows
    con, src = _read(out)
    dups = con.execute(
        f"SELECT count(*) FROM (SELECT user_idx, item_idx FROM {src} "
        f"GROUP BY 1, 2 HAVING count(*) > 1)"
    ).fetchone()[0]
    assert dups == 0, "同一使用者對同一商品不應有多筆互動"


def test_ids_are_dense_and_non_null(built):
    """整數 ID 必須連續且無空值——稀疏或有洞會讓矩陣分解浪費記憶體。"""
    out, stats = built
    con, src = _read(out)
    nulls = con.execute(
        f"SELECT count(*) FROM {src} WHERE user_idx IS NULL OR item_idx IS NULL"
    ).fetchone()[0]
    assert nulls == 0

    max_u, max_i = con.execute(
        f"SELECT max(user_idx), max(item_idx) FROM {src}"
    ).fetchone()
    assert max_u == stats.n_users - 1, "user_idx 應為 0..n_users-1"
    assert max_i == stats.n_items - 1, "item_idx 應為 0..n_items-1"


def test_year_partition_matches_timestamp(built):
    """分區的年份必須與時間戳一致，否則時間切分會取到錯誤的資料。"""
    out, _ = built
    con, src = _read(out)
    bad = con.execute(
        f"SELECT count(*) FROM {src} WHERE year(to_timestamp(ts)) <> year"
    ).fetchone()[0]
    assert bad == 0


def test_ratings_within_valid_range(built):
    out, _ = built
    con, src = _read(out)
    bad = con.execute(
        f"SELECT count(*) FROM {src} WHERE rating NOT BETWEEN 1 AND 5"
    ).fetchone()[0]
    assert bad == 0


def test_timestamps_within_dataset_span(built):
    """資料集聲明為 1996-05 ~ 2023-09；落在範圍外的時間戳會汙染時間切分。"""
    out, _ = built
    con, src = _read(out)
    lo, hi = con.execute(
        f"SELECT to_timestamp(min(ts))::DATE::VARCHAR, "
        f"to_timestamp(max(ts))::DATE::VARCHAR FROM {src}"
    ).fetchone()
    assert lo >= "1996-01-01", f"最早時間戳 {lo} 早於資料集起點"
    assert hi <= "2024-01-01", f"最晚時間戳 {hi} 晚於資料集終點"


def test_mapping_is_deterministic(tmp_path):
    """重跑必須產生完全相同的映射，否則實驗結果無法重現。"""
    con = bi.connect(memory_limit="4GB")
    a = bi.build(con, categories=("Subscription_Boxes",), out_dir=tmp_path / "a")
    b = bi.build(con, categories=("Subscription_Boxes",), out_dir=tmp_path / "b")
    assert (a.n_users, a.n_items, a.kept_rows) == (b.n_users, b.n_items, b.kept_rows)
