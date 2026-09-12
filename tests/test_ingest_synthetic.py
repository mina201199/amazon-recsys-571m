"""無需外部資料，驗證 JSON → 映射 → 去重 → Parquet 的完整管線。"""
import gzip
import json

import duckdb

from amazon_recsys import config
from amazon_recsys.ingest import build_interactions as bi


def test_ingest_synthetic_keeps_earliest_event_and_valid_rows(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(config, "RAW_REVIEWS_DIR", raw)
    rows = [
        {"user_id": "u1", "parent_asin": "p1", "rating": 4,
         "timestamp": 1672531200000, "helpful_vote": 0, "verified_purchase": True},
        {"user_id": "u1", "parent_asin": "p1", "rating": 2,
         "timestamp": 1672617600000, "helpful_vote": 5, "verified_purchase": True},
        {"user_id": "u2", "parent_asin": "p2", "rating": 5,
         "timestamp": 1672617600000, "helpful_vote": 1, "verified_purchase": False},
        {"user_id": "u3", "parent_asin": "p3", "rating": 9,
         "timestamp": 1672617600000, "helpful_vote": 0, "verified_purchase": True},
    ]
    with gzip.open(raw / "Toy.jsonl.gz", "wt", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    output = tmp_path / "interactions"
    with duckdb.connect() as con:
        stats = bi.build(con, categories=("Toy",), out_dir=output,
                         maps_dir=tmp_path / "maps", staging_dir=tmp_path / "staging")
        result = con.execute(
            "SELECT user_idx, item_idx, rating, ts "
            f"FROM read_parquet('{output.as_posix()}/**/*.parquet') "
            "ORDER BY user_idx").fetchall()
    assert stats.kept_rows == 2
    assert stats.n_users == stats.n_items == 2
    assert result == [(0, 0, 4, 1672531200), (1, 1, 5, 1672617600)]
    assert (tmp_path / "maps/user_map.parquet").exists()
