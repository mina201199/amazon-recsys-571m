"""召回評估：各通道、round-robin、加權 RRF，以及可追溯 JSON 紀錄。

使用 valid 決定參數／權重，固定設定後才執行 test。RRF 預設等權不是調參結果。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import numpy as np

from amazon_recsys import config
from amazon_recsys.evaluation import metrics as M
from amazon_recsys.evaluation import splits as S
from amazon_recsys.evaluation.experiment import (
    catalogue_diagnostics,
    channel_reachability,
    provenance,
    public_path,
    save_record,
)
from amazon_recsys.recall import base
from amazon_recsys.recall.als import ALSRecall
from amazon_recsys.recall.category_popularity import CategoryPopularityRecall
from amazon_recsys.recall.content import ContentRecall
from amazon_recsys.recall.covisitation import CoVisitationRecall
from amazon_recsys.recall.popularity import PopularityRecall

CHANNELS = {
    "popularity": lambda k, items: PopularityRecall(
        window_days=90, pool_size=max(2000, k * 4)),
    # 熱度限制在使用者買過的類別內；只需要 --src 既有的 category_idx 欄位。
    "category_popularity": lambda k, items: CategoryPopularityRecall(
        window_days=90, per_category=max(2000, k * 4)),
    "covisitation": lambda k, items: CoVisitationRecall(
        window_days=365, max_items_per_user=20, top_n_neighbours=50, min_cooccurrence=3),
    "als": lambda k, items: ALSRecall(
        factors=32, iterations=10, min_user_interactions=10, min_item_interactions=5),
    # 唯一碰得到冷啟動商品的通道：其餘三路都需要商品已經被買過。
    # 需先執行 scripts/09_build_items.py 產生屬性表，並以 --items 明確指定。
    "content": lambda k, items: ContentRecall(
        items_table=items, per_store=200, per_category=400, cold_slots=50),
}

# 預設不含 category_popularity：它需要 category_idx 欄位，合成展示資料沒有。
# 預設不含 content：它需要一張本腳本無從驗證的外部屬性表。
# 屬性表若與 --src 不是同一份 item_map，item_idx 會指向完全不同的商品，
# 而且不會報錯——因此改為必須以 --items 明確指定，不提供全域路徑回退。
DEFAULT_CHANNELS = ("popularity", "covisitation", "als")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=config.INTERACTIONS_DIR)
    ap.add_argument("--segment", choices=["valid", "test"], default="valid")
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--eval-ks", type=int, nargs="+", default=[10, 100, 500])
    ap.add_argument("--max-users", type=int, default=None, help="合格使用者的嚴格人數上限")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--examples", type=int, default=0, help="另存前 N 位的整數 ID 範例；預設不輸出")
    ap.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                    choices=list(CHANNELS))
    ap.add_argument("--items", type=Path,
                    help="商品屬性表 parquet；--channels 含 content 時必填。"
                         "必須與 --src 出自同一份 item_map")
    ap.add_argument("--weights", type=float, nargs="+", help="依 --channels 順序；只在 valid 調整")
    ap.add_argument("--rrf-constant", type=float, default=60.0)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--memory-limit", default="32GB")
    ap.add_argument("--temp-dir", type=Path, default=config.DATA_ROOT / "duckdb_tmp")
    ap.add_argument("--output", type=Path, help="新 JSON 檔案；拒絕覆寫既有實驗")
    args = ap.parse_args()
    if args.k < 1 or any(k < 1 or k > args.k for k in args.eval_ks):
        ap.error("eval-ks 必須介於 1 與 k 之間")
    if args.max_users is not None and args.max_users < 1:
        ap.error("max-users 必須至少為 1")
    if args.examples < 0:
        ap.error("examples 不可為負")
    if args.bootstrap_samples < 0:
        ap.error("bootstrap-samples 不可為負")
    if len(set(args.channels)) != len(args.channels):
        ap.error("channels 不可重複")
    if "content" in args.channels and args.items is None:
        ap.error("--channels 含 content 時必須指定 --items；"
                 "屬性表與 --src 必須出自同一份 item_map，否則 item_idx 指向不同商品")
    if args.items is not None and not args.items.is_file():
        ap.error(f"找不到商品屬性表：{args.items}")
    weights = args.weights or [1.0] * len(args.channels)
    try:
        base.weighted_rrf([np.empty((0, args.k), dtype=int) for _ in args.channels],
                          args.k, weights, args.rrf_constant)
    except ValueError as exc:
        ap.error(str(exc))
    if not args.src.is_dir() or not any(args.src.rglob("*.parquet")):
        ap.error(f"找不到互動 Parquet：{args.src}；展示請先執行 scripts/07_demo.py")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or config.REPORTS_DIR / "runs" / f"{stamp}.json"
    if output.exists():
        ap.error(f"結果已存在，請選新路徑：{output}")
    root = Path(__file__).resolve().parents[1]
    record = provenance(root, args.src, sys.argv)
    record.update({"status": "running", "parameters": {
        k: public_path(v, root) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "results": {}, "timings_seconds": {}, "channel_parameters": {},
        "protocol": "future-window unseen review interactions; macro user average",
        "weights_note": "user-supplied" if args.weights else "equal weights; not tuned"})
    save_record(output, record)
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={config.N_THREADS}")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        temp = args.temp_dir.as_posix().replace("'", "''")
        con.execute(f"SET temp_directory='{temp}'")
        path = args.src.as_posix().replace("'", "''")
        src = f"read_parquet('{path}/**/*.parquet', hive_partitioning=true)"
        items_table = (
            f"read_parquet('{args.items.as_posix().replace(chr(39), chr(39) * 2)}')"
            if args.items is not None else None
        )
        split = S.DEFAULT_SPLIT
        cutoff = split.feature_cutoff(args.segment)
        record["split"] = {"train_end": S.to_date(split.train_end),
                           "valid_end": S.to_date(split.valid_end),
                           "test_end_exclusive": S.to_date(split.test_end),
                           "feature_cutoff_exclusive": S.to_date(cutoff)}
        t = time.perf_counter()
        users, hists, truths = S.build_eval_set(
            con, src, split, segment=args.segment, max_users=args.max_users, seed=args.seed)
        record["timings_seconds"]["eval_set"] = time.perf_counter() - t
        if not users:
            raise ValueError("沒有合格的評估使用者")
        record["sample"] = {
            "method": "hash top-N among qualified users" if args.max_users else "all qualified",
            "n_users": len(users), "seed": args.seed,
            "user_ids_sha256": hashlib.sha256(json.dumps(users).encode()).hexdigest(),
            "mean_history": sum(map(len, hists)) / len(hists),
            "mean_truth": sum(map(len, truths)) / len(truths),
        }
        record["catalogue"] = catalogue_diagnostics(con, src, cutoff, truths)
        n_items = record["catalogue"]["n_items_before_cutoff"]
        results = {}
        covis_table = None
        for name in args.channels:
            channel = CHANNELS[name](args.k, items_table)
            record["channel_parameters"][name] = {
                f.name: getattr(channel, f.name) for f in fields(channel)
                if not f.name.startswith("_")}
            t = time.perf_counter()
            channel.fit(con, src, cutoff=cutoff)
            record["timings_seconds"][f"{name}_fit"] = time.perf_counter() - t
            t = time.perf_counter()
            results[name] = channel.recommend(hists, k=args.k)
            record["timings_seconds"][f"{name}_recommend"] = time.perf_counter() - t
            evaluated = M.evaluate(results[name], truths, ks=tuple(args.eval_ks), n_items=n_items)
            record["results"][name] = evaluated.metrics
            if hasattr(channel, "stats"):
                record.setdefault("channel_stats", {})[name] = channel.stats()
            if name == "covisitation":
                covis_table = channel._table
            print(f"[{name}] {evaluated}")
            save_record(output, record)
            del channel
        # 答案落在各通道可及範圍內的比例。catalogue_diagnostics 只回答
        # 「答案在訓練期出現過嗎」，這裡再問「在共現圖裡嗎、跨類別嗎」——
        # 兩者要修的東西完全不同，缺口在哪決定了下一步該動哪一層。
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
        if "category_idx" in columns:
            record["reachability"] = channel_reachability(
                con, src, cutoff, truths, hists, covis_table=covis_table)
        else:
            record["reachability"] = {"skipped": "來源缺少 category_idx 欄位"}
        save_record(output, record)
        if len(results) > 1:
            candidates = list(results.values())
            record["union_recall_ceiling"] = float(np.mean([
                len(set().union(*(set(c[u][c[u] >= 0]) for c in candidates)) & truth)
                / len(truth) for u, truth in enumerate(truths)]))
            for name, merge in (
                ("round_robin", lambda: base.merge_channels(candidates, args.k)),
                ("weighted_rrf", lambda: base.weighted_rrf(
                    candidates, args.k, weights, args.rrf_constant)),
            ):
                t = time.perf_counter()
                results[name] = merge()
                record["timings_seconds"][name] = time.perf_counter() - t
                evaluated = M.evaluate(
                    results[name], truths, ks=tuple(args.eval_ks), n_items=n_items
                )
                record["results"][name] = evaluated.metrics
                print(f"[{name}] {evaluated}")
                if "popularity" in results and args.bootstrap_samples:
                    record.setdefault("paired_bootstrap_vs_popularity", {})[name] = {
                        str(k): M.paired_recall_bootstrap(
                            results["popularity"], results[name], truths, k,
                            samples=args.bootstrap_samples, seed=args.seed) for k in args.eval_ks}
        record["history_segments"] = {}
        for label, low, high in (("1-4", 1, 5), ("5-9", 5, 10), ("10+", 10, float("inf"))):
            idx = [u for u, h in enumerate(hists) if low <= len(h) < high]
            if idx:
                record["history_segments"][label] = {"n_users": len(idx), "results": {
                    name: M.evaluate(recs[idx], [truths[u] for u in idx],
                                     ks=tuple(args.eval_ks)).metrics
                    for name, recs in results.items()}}
        if args.examples:
            record["examples"] = [
                {"user_idx": users[u], "history_item_idx": hists[u],
                 "truth_item_idx": sorted(truths[u]),
                 "recommendations": {name: recs[u].tolist() for name, recs in results.items()}}
                for u in range(min(args.examples, len(users)))]
            record["examples_note"] = (
                "Deterministic first N sampled users, not cherry-picked successes. "
                "Integer IDs require the matching source maps; do not infer product names.")
        record["status"] = "completed"
        save_record(output, record)
        print(f"結果已保存：{output}")
        return 0
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_record(output, record)
        raise
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
