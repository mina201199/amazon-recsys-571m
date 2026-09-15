"""把商品 metadata 轉成整數編碼的屬性表（內容式召回用）。

需先執行：uv run python scripts/01_download.py --what meta

用法：
    uv run python scripts/09_build_items.py
    uv run python scripts/09_build_items.py --categories All_Beauty
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amazon_recsys.ingest import build_interactions as bi
from amazon_recsys.ingest import build_items


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--maps-dir", default=None)
    ap.add_argument("--memory-limit", default="32GB")
    args = ap.parse_args()

    con = bi.connect(memory_limit=args.memory_limit)
    st = build_items.build(
        con,
        categories=tuple(args.categories) if args.categories else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        maps_dir=Path(args.maps_dir) if args.maps_dir else None,
    )

    print("=== 商品屬性表 ===")
    print(f"  metadata 列數    {st.raw_rows:,}")
    print(f"  對應到 item_idx  {st.matched_rows:,}  ({st.match_rate:.1%})")
    print(f"  無對應（無互動）  {st.unmatched_rows:,}")
    print(f"  有價格            {st.with_price:,}  "
          f"({st.with_price / max(st.matched_rows, 1):.1%})")
    print(f"  有品牌            {st.with_store:,}  "
          f"({st.with_store / max(st.matched_rows, 1):.1%})")
    print(f"  類別 / 品牌數     {st.n_categories:,} / {st.n_stores:,}")
    print(f"  耗時              {st.seconds:.1f} 秒")
    print("\n  注意：average_rating 與 rating_number 刻意未納入——"
          "它們是 2023-09 的彙總值，含切分點之後的評論。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
