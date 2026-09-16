"""以微型 Parquet 驗證 CLI、實驗保存及失敗狀態。"""
import json
import os
import subprocess
import sys
from pathlib import Path

import duckdb

from amazon_recsys.evaluation.splits import ts

ROOT = Path(__file__).resolve().parents[1]


def test_eval_cli_writes_reproducible_record_and_preserves_existing_output(tmp_path):
    source = tmp_path / "interactions"
    source.mkdir()
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(user_idx INT, item_idx INT, ts BIGINT)")
        con.executemany("INSERT INTO inter VALUES (?, ?, ?)", [
            (1, 1, ts("2023-02-01")), (2, 2, ts("2023-02-01")),
            (1, 2, ts("2023-04-01")), (2, 3, ts("2023-04-01"))])
        dest = (source / "part.parquet").as_posix()
        con.execute(f"COPY inter TO '{dest}' (FORMAT PARQUET)")
    output = tmp_path / "run.json"
    command = [sys.executable, str(ROOT / "scripts/06_recall_eval.py"),
               "--src", str(source), "--k", "2", "--eval-ks", "1", "2",
               "--max-users", "1", "--channels", "popularity", "covisitation",
               "--bootstrap-samples", "20", "--temp-dir", str(tmp_path / "tmp"),
               "--output", str(output)]
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", cwd=ROOT, env=env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "completed"
    assert record["sample"]["n_users"] == 1
    assert set(record["results"]) == {"popularity", "covisitation", "round_robin", "weighted_rrf"}
    assert len(record["code_sha256"]) == 64
    assert record["catalogue"]["n_items_before_cutoff"] == 2
    serialized = json.dumps(record)
    assert str(tmp_path) not in serialized
    assert record["source_directory"].startswith("<external>/")
    previous = output.read_bytes()
    again = subprocess.run(command, capture_output=True, cwd=ROOT, env=env)
    assert again.returncode != 0
    assert output.read_bytes() == previous
    failed_output = tmp_path / "failed.json"
    failure = subprocess.run([
        sys.executable, str(ROOT / "scripts/06_recall_eval.py"), "--src", str(source),
        "--k", "2", "--eval-ks", "2", "--channels", "als",
        "--temp-dir", str(tmp_path / "tmp"), "--output", str(failed_output)],
        capture_output=True, cwd=ROOT, env=env)
    assert failure.returncode != 0
    assert json.loads(failed_output.read_text(encoding="utf-8"))["status"] == "failed"


def _load_eval_script():
    """以模組方式載入評估腳本，才能檢查它的預設值而不必真的跑一次。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_cli", ROOT / "scripts" / "06_recall_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_channels_exclude_content():
    """預設通道不得包含 content——它需要一張本 CLI 無從驗證的外部屬性表。

    content 曾在預設清單裡，且 items_table 綁死在全域設定路徑，
    於是 README 的合成展示指令會靜靜載入正式目錄的商品表。
    """
    module = _load_eval_script()
    assert "content" in module.CHANNELS
    assert "content" not in module.DEFAULT_CHANNELS


def test_content_channel_requires_explicit_items_table(tmp_path):
    """指定 content 卻沒給 --items 時必須當場失敗，不得回退到全域路徑。

    回退是原始 bug 的成因：合成互動表配上正式商品表，item_idx 指向
    完全不同的商品，而且不會有任何錯誤訊息。
    """
    source = tmp_path / "interactions"
    source.mkdir()
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(user_idx INT, item_idx INT, ts BIGINT)")
        con.executemany("INSERT INTO inter VALUES (?, ?, ?)", [
            (1, 1, ts("2023-02-01")), (2, 2, ts("2023-02-01")),
            (1, 2, ts("2023-04-01")), (2, 3, ts("2023-04-01"))])
        con.execute(f"COPY inter TO '{(source / 'part.parquet').as_posix()}' (FORMAT PARQUET)")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/06_recall_eval.py"), "--src", str(source),
         "--k", "2", "--eval-ks", "2", "--channels", "content",
         "--temp-dir", str(tmp_path / "tmp"), "--output", str(tmp_path / "run.json")],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode != 0
    assert "--items" in result.stdout + result.stderr


def test_records_channel_reachability_diagnostics(tmp_path):
    """答案落在各通道可及範圍的診斷必須實際寫進紀錄。

    channel_reachability 早就寫好了，卻沒有任何地方呼叫——而它回答的正是
    目前最該問的問題：Recall@500 只有 3%、理論上限卻有 90%，缺口在哪裡。
    """
    source = tmp_path / "interactions"
    source.mkdir()
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(user_idx INT, item_idx INT, category_idx INT, ts BIGINT)")
        con.executemany("INSERT INTO inter VALUES (?, ?, ?, ?)", [
            (1, 1, 7, ts("2023-02-01")), (2, 2, 7, ts("2023-02-01")),
            (1, 2, 7, ts("2023-04-01")), (2, 3, 8, ts("2023-04-01"))])
        con.execute(f"COPY inter TO '{(source / 'part.parquet').as_posix()}' (FORMAT PARQUET)")
    output = tmp_path / "run.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/06_recall_eval.py"), "--src", str(source),
         "--k", "2", "--eval-ks", "1", "2", "--channels", "popularity", "covisitation",
         "--bootstrap-samples", "20", "--temp-dir", str(tmp_path / "tmp"),
         "--output", str(output)],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    reach = json.loads(output.read_text(encoding="utf-8"))["reachability"]
    assert reach["truth_interactions"] > 0
    assert 0.0 <= reach["same_category_micro"] <= 1.0
    assert reach["same_category_micro"] + reach["cross_category_micro"] == 1.0
    assert 0.0 <= reach["in_covisitation_graph_micro"] <= 1.0


def test_reachability_reports_why_it_was_skipped_without_categories(tmp_path):
    """來源沒有 category_idx 時要明說跳過原因，不能默默不寫。"""
    source = tmp_path / "interactions"
    source.mkdir()
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(user_idx INT, item_idx INT, ts BIGINT)")
        con.executemany("INSERT INTO inter VALUES (?, ?, ?)", [
            (1, 1, ts("2023-02-01")), (2, 2, ts("2023-02-01")),
            (1, 2, ts("2023-04-01")), (2, 3, ts("2023-04-01"))])
        con.execute(f"COPY inter TO '{(source / 'part.parquet').as_posix()}' (FORMAT PARQUET)")
    output = tmp_path / "run.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/06_recall_eval.py"), "--src", str(source),
         "--k", "2", "--eval-ks", "2", "--channels", "popularity",
         "--bootstrap-samples", "0", "--temp-dir", str(tmp_path / "tmp"),
         "--output", str(output)],
        capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert "category_idx" in record["reachability"]["skipped"]
