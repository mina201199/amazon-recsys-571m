"""產生開發用的抽樣資料集（按使用者抽樣，保留完整歷史）。

用法：
    uv run python scripts/05_make_sample.py                 # 預設 5%
    uv run python scripts/05_make_sample.py --fraction 0.2  # 20%
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amazon_recsys import config
from amazon_recsys.ingest import build_interactions as bi
from amazon_recsys.ingest.sample import make_sample


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fraction", type=float, default=0.05, help="抽樣比例（預設 0.05）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", default=None, help="來源互動表目錄")
    ap.add_argument("--out", default=None, help="輸出目錄")
    ap.add_argument("--memory-limit", default="32GB")
    args = ap.parse_args()

    src = Path(args.src) if args.src else config.INTERACTIONS_DIR
    out = Path(args.out) if args.out else config.PARQUET_DIR / "interactions_sample"

    con = bi.connect(memory_limit=args.memory_limit)
    st = make_sample(con, src, out, fraction=args.fraction, seed=args.seed)

    print(f"=== 抽樣結果（目標 {args.fraction:.0%}，seed {args.seed}）===")
    print(f"  使用者    {st.users:>14,} / {st.source_users:,} "
          f"（實際 {st.actual_user_fraction:.2%}）")
    print(f"  互動      {st.interactions:>14,} / {st.source_interactions:,} "
          f"（{st.interactions / max(st.source_interactions, 1):.2%}）")
    print(f"  商品      {st.items:>14,}")
    print(f"  平均每人  {st.interactions / max(st.users, 1):>14.2f} 筆"
          f"   (全量 {st.source_interactions / max(st.source_users, 1):.2f} 筆)")
    print(f"  耗時      {st.seconds:>14.1f} 秒")
    print(f"  大小      {config.human_bytes(_dir_size(out)):>14}")
    print(f"\n  輸出 {out}")
    print(f"  映射表沿用全量的 {config.MAPS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
