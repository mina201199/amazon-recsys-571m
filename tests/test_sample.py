"""抽樣資料集測試。

抽樣最容易犯的錯是按互動列抽，那會把每位使用者的歷史打散，
稀疏性嚴重惡化，於是在抽樣上調出來的參數放到全量就不對——
而且這種偏差不會報錯。這裡的測試就是盯住這件事。
"""

from __future__ import annotations

import duckdb
import pytest

from amazon_recsys.ingest.sample import make_sample


@pytest.fixture
def src(tmp_path):
    """1000 位使用者，每人 10 筆互動，橫跨兩個年份。"""
    d = tmp_path / "full"
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE t AS
        SELECT u AS user_idx, (u * 7 + i) % 500 AS item_idx,
               CAST(4 AS TINYINT) AS rating,
               1600000000 + i * 86400 AS ts,
               CAST(2020 + (i % 2) AS SMALLINT) AS year
        FROM range(1000) AS a(u), range(10) AS b(i)
    """)
    con.execute(f"""
        COPY t TO '{d.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)
    return con, d


def test_samples_users_not_rows(src, tmp_path):
    """被選中的使用者必須保留全部 10 筆歷史，不是被抽掉 95%。

    這是整個模組存在的理由：按列抽樣會讓平均歷史長度崩潰。
    """
    con, d = src
    st = make_sample(con, d, tmp_path / "s", fraction=0.2)
    per_user = st.interactions / st.users
    assert per_user == pytest.approx(10.0), (
        f"每人平均剩 {per_user:.1f} 筆，應為 10 筆——歷史被打散了"
    )


def test_fraction_is_approximately_honoured(src, tmp_path):
    con, d = src
    st = make_sample(con, d, tmp_path / "s", fraction=0.2)
    assert 0.15 < st.actual_user_fraction < 0.25


def test_sampling_is_deterministic(src, tmp_path):
    """同樣的 seed 必須抽到同樣的使用者，否則實驗無法重現。"""
    con, d = src
    a = make_sample(con, d, tmp_path / "a", fraction=0.2, seed=7)
    b = make_sample(con, d, tmp_path / "b", fraction=0.2, seed=7)
    assert (a.users, a.interactions) == (b.users, b.interactions)

    users_a = con.execute(
        f"SELECT DISTINCT user_idx FROM read_parquet('{(tmp_path / 'a').as_posix()}/**/*.parquet')"
    ).fetchall()
    users_b = con.execute(
        f"SELECT DISTINCT user_idx FROM read_parquet('{(tmp_path / 'b').as_posix()}/**/*.parquet')"
    ).fetchall()
    assert sorted(users_a) == sorted(users_b)


def test_different_seeds_pick_different_users(src, tmp_path):
    con, d = src
    make_sample(con, d, tmp_path / "a", fraction=0.2, seed=1)
    make_sample(con, d, tmp_path / "b", fraction=0.2, seed=2)
    ua = {r[0] for r in con.execute(
        f"SELECT DISTINCT user_idx FROM read_parquet('{(tmp_path / 'a').as_posix()}/**/*.parquet')"
    ).fetchall()}
    ub = {r[0] for r in con.execute(
        f"SELECT DISTINCT user_idx FROM read_parquet('{(tmp_path / 'b').as_posix()}/**/*.parquet')"
    ).fetchall()}
    assert ua != ub


def test_ids_are_not_re_encoded(src, tmp_path):
    """抽樣沿用全量的映射表，所以 ID 不可被重新編碼。

    若這裡改成重新編碼成 0..n-1，抽樣與全量的 item_idx 就會指向
    不同商品，兩邊的實驗結果再也無法對照。
    """
    con, d = src
    make_sample(con, d, tmp_path / "s", fraction=0.2)
    mx = con.execute(
        f"SELECT max(user_idx) FROM read_parquet('{(tmp_path / 's').as_posix()}/**/*.parquet')"
    ).fetchone()[0]
    # 1000 位使用者抽 20%，若重新編碼則 max 會接近 200；沿用原 ID 則接近 999
    assert mx > 500, "ID 似乎被重新編碼了"


def test_year_partitions_are_preserved(src, tmp_path):
    con, d = src
    make_sample(con, d, tmp_path / "s", fraction=0.5)
    parts = sorted(p.name for p in (tmp_path / "s").iterdir() if p.is_dir())
    assert parts == ["year=2020", "year=2021"]


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
def test_invalid_fraction_rejected(src, tmp_path, bad):
    con, d = src
    with pytest.raises(ValueError, match="fraction"):
        make_sample(con, d, tmp_path / "s", fraction=bad)
