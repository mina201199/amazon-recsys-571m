"""把原始 .jsonl.gz 轉成整數編碼、按年分區的互動 Parquet 表。

用法：
    uv run python scripts/02_build_interactions.py                  # 全量 34 個類別
    uv run python scripts/02_build_interactions.py --categories All_Beauty Gift_Cards
    uv run python scripts/02_build_interactions.py --out-dir D:/tmp/test  # 測試用
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amazon_recsys import config  # noqa: E402
from amazon_recsys.ingest import build_interactions as bi  # noqa: E402


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="+", default=None, help="只處理指定類別")
    ap.add_argument("--out-dir", default=None, help="輸出目錄（預設為設定檔中的路徑）")
    ap.add_argument("--memory-limit", default="48GB", help="DuckDB 記憶體上限")
    args = ap.parse_args()

    config.ensure_dirs()
    cats = tuple(args.categories) if args.categories else None
    out = Path(args.out_dir) if args.out_dir else config.INTERACTIONS_DIR

    src_bytes = sum(
        (config.RAW_REVIEWS_DIR / f"{c}.jsonl.gz").stat().st_size
        for c in (cats or config.CATEGORIES)
        if (config.RAW_REVIEWS_DIR / f"{c}.jsonl.gz").exists()
    )

    print(f"處理 {len(cats or config.CATEGORIES)} 個類別，"
          f"原始大小 {config.human_bytes(src_bytes)}")
    print(f"執行緒 {config.N_THREADS}，記憶體上限 {args.memory_limit}\n")

    con = bi.connect(memory_limit=args.memory_limit)
    st = bi.build(con, categories=cats, out_dir=out)

    out_bytes = _dir_size(out)
    print("=== 轉檔結果 ===")
    print(f"  解析列數        {st.raw_rows:,}")
    print(f"  去重後列數      {st.kept_rows:,} "
          f"(移除 {st.raw_rows - st.kept_rows:,}，"
          f"{100 * (st.raw_rows - st.kept_rows) / max(st.raw_rows, 1):.2f}%)")
    print(f"  使用者          {st.n_users:,}")
    print(f"  商品            {st.n_items:,}")
    print(f"  類別            {st.n_categories}")
    print(f"  時間範圍        {st.first_date} ~ {st.last_date}")
    print(f"  耗時            {st.seconds:.1f} 秒 "
          f"({st.raw_rows / max(st.seconds, 0.001):,.0f} 列/秒)")
    print()
    print("=== 壓縮效果 ===")
    print(f"  原始 .jsonl.gz  {config.human_bytes(src_bytes)}")
    print(f"  Parquet 輸出    {config.human_bytes(out_bytes)}")
    if out_bytes:
        print(f"  壓縮比          {src_bytes / out_bytes:.1f}x")
        print(f"  每列位元組      {out_bytes / max(st.kept_rows, 1):.1f} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
