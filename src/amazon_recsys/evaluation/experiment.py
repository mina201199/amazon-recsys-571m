"""實驗來源、資料目錄指紋與時間點一致的覆蓋診斷。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa


def public_path(path: Path, root: Path) -> str:
    """避免公開實驗紀錄洩漏 Windows 使用者名稱或磁碟目錄。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def provenance(root: Path, source: Path, argv: list[str]) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", *args],
                cwd=root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = sorted(source.rglob("*.parquet"))
    inventory = [{"path": p.relative_to(source).as_posix(), "bytes": p.stat().st_size,
                  "mtime_ns": p.stat().st_mtime_ns} for p in files]
    code_files = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py"),
                         root / "uv.lock", root / "pyproject.toml"])
    code = hashlib.sha256()
    for p in code_files:
        if p.exists():
            code.update(p.relative_to(root).as_posix().encode())
            code.update(p.read_bytes())
    versions = {}
    for package in ("duckdb", "numpy", "scipy", "implicit", "pyarrow", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    path_flags = {"--src", "--output", "--temp-dir"}
    safe_argv = []
    hide_next = False
    for token in argv:
        if hide_next:
            safe_argv.append(f"<path>/{Path(token).name}")
            hide_next = False
        else:
            safe_argv.append(token)
            hide_next = token in path_flags
    return {
        "created_utc": datetime.now(UTC).isoformat(), "argv": safe_argv,
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "code_sha256": code.hexdigest(), "versions": versions,
        "python": platform.python_version(), "platform": platform.platform(),
        "source_directory": public_path(source, root), "source_files": inventory,
        "source_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        "source_fingerprint_note": "檔案清單、大小與修改時間指紋；不是資料內容 SHA256。",
    }


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def catalogue_diagnostics(con, src: str, cutoff: int, truths: list[set[int]]) -> dict:
    """全目錄只作資料描述；coverage 分母以 cutoff 前可見商品定義。

    未來首次出現的商品保留在答案中，另外量測不可達比例，不偷偷移除。
    """
    con.execute(f"""CREATE OR REPLACE TEMP TABLE train_catalogue AS
        SELECT DISTINCT item_idx FROM {src} WHERE ts < {cutoff}""")
    n_train = con.execute("SELECT count(*) FROM train_catalogue").fetchone()[0]
    n_all = con.execute(f"SELECT count(DISTINCT item_idx) FROM {src}").fetchone()[0]
    table = pa.table({
        "row_id": pa.array([u for u, truth in enumerate(truths) for _ in truth], pa.int64()),
        "item_idx": pa.array([i for truth in truths for i in sorted(truth)], pa.int64()),
    })
    con.register("eval_truth_arrow", table)
    try:
        total, cold, ceiling = con.execute("""
            WITH by_user AS (
                SELECT row_id, count(*) AS n,
                       count(c.item_idx) AS reachable
                FROM eval_truth_arrow t LEFT JOIN train_catalogue c USING (item_idx)
                GROUP BY row_id
            ) SELECT sum(n), sum(n - reachable), avg(reachable::DOUBLE / n) FROM by_user
        """).fetchone()
    finally:
        con.unregister("eval_truth_arrow")
    return {"n_items_before_cutoff": n_train, "n_items_all_dates": n_all,
            "truth_interactions": int(total), "cold_truth_interactions": int(cold),
            "cold_truth_fraction_micro": cold / total,
            "train_catalogue_recall_ceiling_macro": ceiling,
            "coverage_denominator": "distinct items with ts < feature_cutoff"}


def channel_reachability(
    con, src: str, cutoff: int, truths: list[set[int]],
    histories: list[list[int]], covis_table: str | None = None,
) -> dict:
    """答案落在各通道可及範圍內的比例，補 catalogue_diagnostics 未涵蓋的部分。

    catalogue_diagnostics 回答的是「答案在訓練期出現過嗎」（理論上限）。
    這裡再往下問兩題，因為兩者要修的東西完全不同：

    * **在共現圖裡嗎** —— 若答案大多不在圖內，共現通道無論權重怎麼調
      都撈不到，該修的是圖的覆蓋率而非融合。
    * **跨類別嗎** —— 使用者跳到從未買過的類別時，共現與 ALS 都只能
      靠間接關聯，是已知的弱項。

    兩者皆以互動為單位（micro），與 catalogue_diagnostics 的 micro 欄位可比。
    """
    rows_u = [u for u, truth in enumerate(truths) for _ in truth]
    rows_i = [i for truth in truths for i in sorted(truth)]
    if not rows_i:
        return {"truth_interactions": 0}

    con.register("reach_truth_arrow", pa.table({
        "row_id": pa.array(rows_u, pa.int64()),
        "item_idx": pa.array(rows_i, pa.int64()),
    }))
    hist_u = [u for u, h in enumerate(histories) for _ in h]
    hist_i = [int(i) for h in histories for i in h]
    con.register("reach_hist_arrow", pa.table({
        "row_id": pa.array(hist_u, pa.int64()),
        "item_idx": pa.array(hist_i, pa.int64()),
    }))
    try:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE reach_hist_cat AS
            SELECT DISTINCT h.row_id, s.category_idx
            FROM reach_hist_arrow h
            JOIN (SELECT DISTINCT item_idx, category_idx FROM {src}) s USING (item_idx)""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE reach_truth_cat AS
            SELECT t.row_id, t.item_idx, s.category_idx
            FROM reach_truth_arrow t
            JOIN (SELECT DISTINCT item_idx, category_idx FROM {src}) s USING (item_idx)""")
        total, same_cat = con.execute("""
            SELECT count(*),
                   count(*) FILTER (WHERE EXISTS (
                       SELECT 1 FROM reach_hist_cat h
                       WHERE h.row_id = t.row_id AND h.category_idx = t.category_idx))
            FROM reach_truth_cat t
        """).fetchone()
        in_graph = None
        if covis_table:
            in_graph = con.execute(f"""
                SELECT count(*) FROM reach_truth_arrow t
                WHERE EXISTS (SELECT 1 FROM {covis_table} c WHERE c.dst = t.item_idx)
            """).fetchone()[0]
    finally:
        con.unregister("reach_truth_arrow")
        con.unregister("reach_hist_arrow")

    out = {
        "truth_interactions": int(total),
        "same_category_micro": same_cat / total if total else None,
        "cross_category_micro": (total - same_cat) / total if total else None,
    }
    if in_graph is not None:
        out["in_covisitation_graph_micro"] = in_graph / len(rows_i)
    return out
