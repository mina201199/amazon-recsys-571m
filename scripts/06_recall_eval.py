"""量測召回層：各通道與合併後的 Recall@K，並記錄每個環節的耗時。

召回層的 Recall@K 是整個系統的天花板——召回沒撈到的商品，
排序層再強也救不回來。所以這個數字要先做高，再去調排序。

本腳本同時記錄各環節耗時，用來判斷全量迭代是否可行、
是否需要改用抽樣資料開發。

用法：
    uv run python scripts/06_recall_eval.py
    uv run python scripts/06_recall_eval.py --k 500 --max-users 50000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from amazon_recsys import config
from amazon_recsys.evaluation import metrics as M
from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall import base
from amazon_recsys.recall.als import ALSRecall
from amazon_recsys.recall.covisitation import CoVisitationRecall
from amazon_recsys.recall.popularity import PopularityRecall

CHANNELS = {
    "popularity": lambda k: PopularityRecall(window_days=90, pool_size=max(2000, k * 4)),
    "covisitation": lambda k: CoVisitationRecall(
        window_days=730, max_items_per_user=50, top_n_neighbours=100, min_cooccurrence=2
    ),
    "als": lambda k: ALSRecall(factors=64, iterations=15, min_user_interactions=5),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=None)
    ap.add_argument("--segment", choices=["valid", "test"], default="valid",
                    help="預設用 valid；test 段保留到最後只評估一次")
    ap.add_argument("--k", type=int, default=500, help="召回候選數")
    ap.add_argument("--eval-ks", type=int, nargs="+", default=[10, 100, 500])
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS),
                    choices=list(CHANNELS))
    ap.add_argument("--memory-limit", default="32GB")
    args = ap.parse_args()

    src_dir = Path(args.src) if args.src else config.INTERACTIONS_DIR
    src = f"read_parquet('{src_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"

    con = duckdb.connect()
    con.execute(f"SET threads={config.N_THREADS}")
    con.execute(f"SET memory_limit='{args.memory_limit}'")
    tmp = config.DATA_ROOT / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    split = S.DEFAULT_SPLIT
    cutoff = split.feature_cutoff(args.segment)
    print(f"切分：{split.describe()}")
    print(f"評估段：{args.segment}（特徵只能用到 {S.to_date(cutoff)} 之前）\n")

    t = time.time()
    users, hists, truths = S.build_eval_set(
        con, src, split, segment=args.segment, max_users=args.max_users
    )
    t_eval_set = time.time() - t
    if not users:
        print("沒有合格的使用者。")
        return 1

    hl = sorted(len(h) for h in hists)
    print(f"評估集：{len(users):,} 位使用者（建構耗時 {t_eval_set:.1f}s）")
    print(f"  歷史長度 中位數 {hl[len(hl) // 2]}，平均 {sum(hl) / len(hl):.1f}，最長 {hl[-1]}")
    print(f"  正確答案 平均 {sum(len(x) for x in truths) / len(truths):.2f}\n")

    n_items = con.execute(f"SELECT count(DISTINCT item_idx) FROM {src}").fetchone()[0]

    results, timings = {}, {"評估集建構": t_eval_set}
    for name in args.channels:
        ch = CHANNELS[name](args.k)
        t = time.time()
        ch.fit(con, src, cutoff=cutoff)
        t_fit = time.time() - t

        t = time.time()
        recs = ch.recommend(hists, k=args.k)
        t_rec = time.time() - t

        res = M.evaluate(recs, truths, ks=tuple(args.eval_ks), n_items=n_items)
        results[name] = (recs, res)
        timings[f"{name} fit"] = t_fit
        timings[f"{name} recommend"] = t_rec

        extra = ""
        if hasattr(ch, "stats"):
            extra = "  " + "  ".join(f"{k}={v:,}" for k, v in ch.stats().items())
        print(f"[{name}] fit {t_fit:6.1f}s  recommend {t_rec:6.1f}s{extra}")
        print(f"         {res}")

    if len(results) > 1:
        t = time.time()
        merged = base.merge_channels([r[0] for r in results.values()], k=args.k)
        t_merge = time.time() - t
        timings["合併"] = t_merge
        res = M.evaluate(merged, truths, ks=tuple(args.eval_ks), n_items=n_items)
        print(f"\n[合併 {len(results)} 路] {t_merge:.1f}s")
        print(f"         {res}")

    print("\n=== 耗時明細 ===")
    total = sum(timings.values())
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"  {k:<24} {v:8.1f}s  ({100 * v / total:4.1f}%)")
    print(f"  {'合計':<24} {total:8.1f}s  ({total / 60:.1f} 分鐘)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
