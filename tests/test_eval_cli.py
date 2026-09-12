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
