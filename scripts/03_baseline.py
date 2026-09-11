"""跑熱門商品基線，產出第一個可比較的數字。

這條基線是後續所有模型必須打敗的對手。用法：

    uv run python scripts/03_baseline.py
    uv run python scripts/03_baseline.py --src D:/tmp/test_interactions --max-users 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from amazon_recsys import config
from amazon_recsys.evaluation import metrics as M
from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall.popularity import PopularityRecall


def main() -> int:
    # 重導向到檔案時 stdout 會緩衝，長時間執行就看不到進度
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None, help="互動表 Parquet 目錄")
    ap.add_argument("--segment", choices=["valid", "test"], default="test")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 100, 500])
    ap.add_argument("--max-users", type=int, default=None, help="抽樣使用者數上限")
    ap.add_argument(
        "--window-days", type=int, default=90,
        help="熱度統計的時間視窗天數（0 表示使用全期）",
    )
    args = ap.parse_args()

    src_dir = Path(args.src) if args.src else config.INTERACTIONS_DIR
    src = (
        f"read_parquet('{src_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    )

    con = duckdb.connect()
    con.execute(f"SET threads={config.N_THREADS}")
    split = S.DEFAULT_SPLIT
    print(f"切分：{split.describe()}")
    print(f"評估段：{args.segment}（特徵只能用到 "
          f"{S.to_date(split.feature_cutoff(args.segment))} 之前）\n")

    print("=== 各段規模 ===")
    for row in S.split_summary(con, src, split):
        print(f"  {row['segment']:6s} 互動 {row['interactions']:>12,}  "
              f"使用者 {row['users']:>11,}  商品 {row['items']:>10,}")

    print("\n=== 建立評估集 ===")
    users, hists, truths = S.build_eval_set(
        con, src, split, segment=args.segment, max_users=args.max_users
    )
    if not users:
        print("  沒有合格的使用者（需同時具備歷史與該段的新商品互動）。")
        return 1

    hist_len = [len(h) for h in hists]
    print(f"  合格使用者 {len(users):,}")
    print(f"  歷史長度   中位數 {sorted(hist_len)[len(hist_len) // 2]}，"
          f"平均 {sum(hist_len) / len(hist_len):.1f}，最長 {max(hist_len)}")
    print(f"  正確答案數 平均 {sum(len(t) for t in truths) / len(truths):.2f}")

    n_items = con.execute(f"SELECT count(DISTINCT item_idx) FROM {src}").fetchone()[0]

    print("\n=== 熱門商品基線 ===")
    k_max = max(args.ks)
    rec = PopularityRecall(
        window_days=args.window_days or None,
        pool_size=max(2_000, k_max * 4),
    )
    rec.fit(con, src, cutoff=split.feature_cutoff(args.segment))
    print(f"  候選池 {len(rec._ranked):,} 個商品"
          f"（視窗 {args.window_days or '全期'} 天）")

    recs = rec.recommend(hists, k=k_max)
    result = M.evaluate(recs, truths, ks=tuple(args.ks), n_items=n_items)

    print(f"\n  評估 {result.n_users:,} 位使用者，商品目錄 {n_items:,}")
    for k in args.ks:
        print(f"    Recall@{k:<4} {result.metrics[f'recall@{k}']:.5f}    "
              f"NDCG@{k:<4} {result.metrics[f'ndcg@{k}']:.5f}")
    print(f"    Coverage    {result.metrics['coverage']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
