"""產生開發用的抽樣資料集。

全量 5.71 億列的每次實驗都要數十分鐘，調參與除錯不該付這個代價。
標準做法是用抽樣快速迭代，確認方法正確後才跑全量。

## 為什麼按使用者抽樣，而不是按互動列

隨機抽 5% 的互動列會把每位使用者的歷史打散到剩 5%：
平均 10.5 筆的歷史變成 0.5 筆。稀疏性嚴重惡化，抽樣上得到的結論
完全無法推廣到全量——而這種偏差不會報錯，只會讓你在錯的資料上
調出錯的參數。

按使用者抽樣則保留每個人的完整歷史，per-user 的分布與全量一致。

## 已知的限制

商品端的共現會變稀疏：每個商品的互動者少了 20 倍，共現配對數
大約掉 400 倍（平方關係）。因此抽樣資料**適合**驗證流程是否正確、
比較模型的相對優劣，**不適合**用來報告最終數字，也不適合調整
`min_cooccurrence` 這類直接跟絕對次數有關的參數。

## 映射表沿用全量的

抽樣資料不重新編碼 ID，直接沿用全量的映射表。這樣 item_idx=65404
在抽樣與全量裡指的是同一個商品，兩邊的實驗結果可以直接對照。
代價是 ID 不再連續（0..n-1 有空洞），但召回層的實作都會自行壓縮索引，
不受影響。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

# 抽樣用的乘數：Knuth 建議的黃金比例常數，讓連續的 user_idx 打散得更均勻
_MIX = 2_654_435_761
_SCALE = 1_000_000


@dataclass
class SampleStats:
    fraction: float
    users: int
    interactions: int
    items: int
    source_users: int
    source_interactions: int
    seconds: float

    @property
    def actual_user_fraction(self) -> float:
        return self.users / max(self.source_users, 1)


def make_sample(
    con: duckdb.DuckDBPyConnection,
    src_dir: Path,
    out_dir: Path,
    fraction: float = 0.05,
    seed: int = 42,
) -> SampleStats:
    """抽取 `fraction` 比例的使用者，保留他們的全部互動。

    抽樣是決定性的：同樣的 seed 必定得到同樣的使用者集合，
    實驗才能重現。
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction 必須介於 0 與 1 之間，收到 {fraction}")

    t0 = time.time()
    src = f"read_parquet('{src_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_n, src_u = con.execute(
        f"SELECT count(*), count(DISTINCT user_idx) FROM {src}"
    ).fetchone()

    threshold = int(fraction * _SCALE)
    keep = f"hash(user_idx * {_MIX} + {seed}) % {_SCALE} < {threshold}"

    con.execute(f"""
        COPY (SELECT * FROM {src} WHERE {keep})
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)

    out = f"read_parquet('{out_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    n, u, i = con.execute(
        f"SELECT count(*), count(DISTINCT user_idx), count(DISTINCT item_idx) FROM {out}"
    ).fetchone()

    return SampleStats(
        fraction=fraction,
        users=u,
        interactions=n,
        items=i,
        source_users=src_u,
        source_interactions=src_n,
        seconds=time.time() - t0,
    )
