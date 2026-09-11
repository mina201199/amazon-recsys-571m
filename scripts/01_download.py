"""下載 Amazon Reviews 2023 原始資料。

用法：
    uv run python scripts/01_download.py --what reviews     # 71.2 GB，約 2.9 小時
    uv run python scripts/01_download.py --what meta        # 24.5 GB，約 1 小時
    uv run python scripts/01_download.py --what all         # 兩者都下載

中斷後直接重跑即可從斷點續傳。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amazon_recsys import config
from amazon_recsys.ingest import download


def main() -> int:
    # 重導向到檔案時 stdout 會緩衝，長時間執行就看不到進度
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--what", choices=["reviews", "meta", "all"], default="reviews",
        help="要下載哪一批（預設 reviews）",
    )
    ap.add_argument(
        "--workers", type=int, default=config.DOWNLOAD_WORKERS,
        help=f"並行連線數（預設 {config.DOWNLOAD_WORKERS}，實測此值已達頻寬飽和）",
    )
    ap.add_argument(
        "--categories", nargs="+", default=None,
        help="只下載指定類別（預設全部 34 個）",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="只查詢大小與現況，不實際下載",
    )
    args = ap.parse_args()

    config.ensure_dirs()
    cats = tuple(args.categories) if args.categories else config.CATEGORIES

    specs: list[download.FileSpec] = []
    if args.what in ("reviews", "all"):
        specs += download.review_specs(cats)
    if args.what in ("meta", "all"):
        specs += download.meta_specs(cats)

    if args.dry_run:
        import httpx

        print(f"{'檔案':<40} {'遠端':>12} {'本機':>12}  狀態")
        total = have = 0
        with httpx.Client(timeout=30.0) as c:
            for s in specs:
                r = download.remote_size(c, s.url) or 0
                lo = s.dest.stat().st_size if s.dest.exists() else 0
                total += r
                have += min(lo, r)
                state = "完成" if lo == r and r else ("部分" if lo else "未下載")
                print(
                    f"{s.name:<40} {config.human_bytes(r):>12} "
                    f"{config.human_bytes(lo):>12}  {state}"
                )
        print(
            f"\n總計 {config.human_bytes(total)}，"
            f"已有 {config.human_bytes(have)}，"
            f"待下載 {config.human_bytes(total - have)}"
        )
        return 0

    t0 = time.time()
    results = download.download_all(specs, workers=args.workers)
    download.summarise(results)

    elapsed = time.time() - t0
    moved = sum(r.bytes_written for r in results)
    print(f"\n耗時 {elapsed / 60:.1f} 分鐘", end="")
    if elapsed > 0 and moved > 0:
        print(f"，平均 {config.human_bytes(moved / elapsed)}/s")
    else:
        print()

    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
