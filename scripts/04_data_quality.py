"""產出資料品質與分布報告。

用法：
    uv run python scripts/04_data_quality.py
    uv run python scripts/04_data_quality.py --src D:/tmp/test --out reports/test.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from amazon_recsys import config
from amazon_recsys.evaluation import quality, splits


def main() -> int:
    # 重導向到檔案時 stdout 會緩衝，長時間執行就看不到進度
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None, help="互動表 Parquet 目錄")
    ap.add_argument("--out", default=None, help="報告輸出路徑（.md）")
    ap.add_argument("--memory-limit", default="48GB")
    args = ap.parse_args()

    src_dir = Path(args.src) if args.src else config.INTERACTIONS_DIR
    out = Path(args.out) if args.out else config.REPORTS_DIR / "data_quality.md"
    src = f"read_parquet('{src_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"

    con = duckdb.connect()
    con.execute(f"SET threads={config.N_THREADS}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    tmp = config.DATA_ROOT / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    report = quality.build_report(con, src, splits.DEFAULT_SPLIT)
    print(report.text())

    quality.write_report(report, out)
    print(f"\n報告已寫入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
