"""把原始 .jsonl.gz 轉成整數編碼、按年分區的互動表。

這是整條管線的地基。設計重點（理由見 docs/specs/）：

1. **用 parent_asin 而非 asin**：顏色/尺寸變體共用 parent_asin。
   用 asin 會虛增商品數並稀釋每個商品的互動訊號。
2. **ID 整數編碼**：user_id 是 28 字元字串，571M 列光存 ID 要 ~16 GB；
   映射成 int32 後降到 2.3 GB，讓全量表能常駐記憶體。
3. **按年分區**：評估要按時間切，年分區讓查詢只掃必要檔案。

## 為什麼分三階段，而不是一口氣做完

最初的版本把解析結果整個存成記憶體暫存表，再跑去重。實測發現
該表每列佔 85.5 bytes（字串 ID 是主因），全量 5.71 億列就是 **45.5 GB**,
幾乎吃光 48 GB 的記憶體預算。後續每個運算只剩 2.5 GB 可用，
全部溢寫到磁碟，跑了 75 分鐘仍未完成。

改成中間用 Parquet 落地後，每個階段的記憶體用量都有明確上界：

    階段 1  JSON → staging Parquet   COPY 為串流寫出，用量固定
    階段 2  staging → 三張映射表      Parquet 是欄式的，只讀需要的那一欄
    階段 3  join + 去重 → 分區 Parquet  鍵已是 int32

階段 3 用 GROUP BY 而非視窗函數。單看速度，視窗函數其實較快
（2950 萬列實測 3.6s vs 9.3s），但它需要對**輸入**做全域排序，
記憶體上界是輸入的大小；GROUP BY 的雜湊表只跟**輸出**的列數成正比。
在輸入遠大於記憶體時，後者才跑得完。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from amazon_recsys import config

# 保留的欄位。刻意丟棄：
#   images  — 巢狀結構，推薦模型用不到
#   title   — 評論標題，本專案不做 NLP
#   text    — 平均 173 字元，全量約 99 GB，不做 NLP 就是純負擔
#   asin    — 改用 parent_asin，見模組說明

# 資料集聲明的時間範圍（1996-01 ~ 2024-01）的毫秒時間戳。
# 實測有少量離群值，放著會汙染時間切分。
_TS_MIN, _TS_MAX = 820_454_400_000, 1_704_067_200_000


@dataclass
class IngestStats:
    raw_rows: int
    kept_rows: int
    n_users: int
    n_items: int
    n_categories: int
    first_date: str
    last_date: str
    seconds: float
    stage_seconds: dict[str, float]


def connect(memory_limit: str = "48GB") -> duckdb.DuckDBPyConnection:
    """建立設定好的 DuckDB 連線。

    記憶體上限留餘裕給作業系統；超出部分溢寫到 D 槽的暫存目錄
    （不可用 C 槽，那是系統碟且空間較少）。
    """
    con = duckdb.connect()
    tmp = config.DATA_ROOT / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={config.N_THREADS}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")  # 允許重排以節省記憶體
    return con


def _sources(categories: tuple[str, ...] | None) -> list[str]:
    """要讀取的檔案清單。只納入實際存在的檔案。"""
    cats = categories or config.CATEGORIES
    paths = [config.RAW_REVIEWS_DIR / f"{c}.jsonl.gz" for c in cats]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"缺少 {len(missing)} 個檔案：{missing[:5]}")
    return [p.as_posix() for p in paths]


def build(
    con: duckdb.DuckDBPyConnection,
    categories: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    maps_dir: Path | None = None,
    staging_dir: Path | None = None,
    keep_staging: bool = False,
) -> IngestStats:
    """執行完整轉檔流程，回傳統計數字與各階段耗時。

    映射表預設寫到 `out_dir` 的同層 `maps/`，而非固定位置。
    這點攸關正確性：整數 ID 只有搭配產生它的那份映射表才有意義。
    若互動表與映射表來自不同批資料，item_idx 會被解碼成完全不同的
    商品——而且不會拋出任何錯誤，只會安靜地給出錯的結果。

    `staging_dir` 是階段 1 的中間產物；預設在完成後刪除，
    傳 `keep_staging=True` 可保留以便除錯或重跑階段 2-3。
    """
    t0 = time.time()
    out_dir = out_dir or config.INTERACTIONS_DIR
    maps_dir = maps_dir or (out_dir.parent / "maps")
    staging = staging_dir or (out_dir.parent / "_staging")
    for d in (out_dir, maps_dir, staging.parent):
        d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)

    files = _sources(categories)
    files_sql = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    stage_s: dict[str, float] = {}

    # --- 階段 1：串流解析 JSON → staging Parquet -------------------------
    # 直接 COPY 而不建記憶體暫存表：COPY 是串流寫出，記憶體用量固定。
    # filename=true 讓我們能從檔名還原類別。
    t = time.time()
    con.execute(f"""
        COPY (
            SELECT
                user_id,
                parent_asin,
                CAST(rating AS TINYINT)                          AS rating,
                CAST(timestamp / 1000 AS INTEGER)                AS ts,
                CAST(LEAST(helpful_vote, 32767) AS SMALLINT)     AS helpful_vote,
                verified_purchase,
                replace(parse_filename(filename), '.jsonl.gz', '')  AS category
            FROM read_json_auto({files_sql}, ignore_errors=true, filename=true)
            WHERE user_id      IS NOT NULL
              AND parent_asin  IS NOT NULL
              AND timestamp    IS NOT NULL
              AND rating       IS NOT NULL
              AND timestamp BETWEEN {_TS_MIN} AND {_TS_MAX}
        ) TO '{staging.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true,
         OVERWRITE_OR_IGNORE)
    """)
    stage_s["1_解析"] = time.time() - t
    # PER_THREAD_OUTPUT 讓每個執行緒各寫一個檔案，寫入與後續讀取都能平行
    src = f"read_parquet('{staging.as_posix()}/*.parquet')"
    raw_rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    # --- 階段 2：建立整數映射表 ------------------------------------------
    # 從 Parquet 讀取，欄式格式讓 DISTINCT 只掃需要的那一欄。
    # ORDER BY 原始 ID → 重跑結果完全一致，可重現。
    t = time.time()
    for name, col, idx in (
        ("user_map", "user_id", "user_idx"),
        ("item_map", "parent_asin", "item_idx"),
        ("category_map", "category", "category_idx"),
    ):
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT {col},
                   CAST(row_number() OVER (ORDER BY {col}) - 1 AS INTEGER) AS {idx}
            FROM (SELECT DISTINCT {col} FROM {src})
        """)
        con.execute(f"""
            COPY {name} TO '{(maps_dir / (name + ".parquet")).as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    stage_s["2_映射表"] = time.time() - t

    n_users = con.execute("SELECT count(*) FROM user_map").fetchone()[0]
    n_items = con.execute("SELECT count(*) FROM item_map").fetchone()[0]
    n_cats = con.execute("SELECT count(*) FROM category_map").fetchone()[0]

    # --- 階段 3：join + 去重 + 寫出分區表 --------------------------------
    # 同一使用者對同一商品可能有多筆（重新評論）。推薦系統只需要互動事實，
    # 保留最早那筆——最早的互動才是「他決定買」的時間點。
    #
    # 用 GROUP BY 而非視窗函數：單看速度視窗函數較快，但它的記憶體上界是
    # 輸入大小；GROUP BY 的雜湊表只跟去重後的列數成正比。輸入遠大於記憶體時，
    # 只有後者跑得完。
    t = time.time()
    con.execute(f"""
        COPY (
            SELECT *, CAST(year(to_timestamp(ts)) AS SMALLINT) AS year
            FROM (
                SELECT
                    u.user_idx,
                    i.item_idx,
                    arg_min(r.rating, r.ts)                       AS rating,
                    min(r.ts)                                     AS ts,
                    arg_min(r.helpful_vote, r.ts)                 AS helpful_vote,
                    arg_min(r.verified_purchase, r.ts)            AS verified_purchase,
                    CAST(arg_min(c.category_idx, r.ts) AS TINYINT) AS category_idx
                FROM {src} r
                JOIN user_map     u USING (user_id)
                JOIN item_map     i USING (parent_asin)
                JOIN category_map c USING (category)
                GROUP BY u.user_idx, i.item_idx
            )
        ) TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)
    stage_s["3_去重與寫出"] = time.time() - t

    final = f"read_parquet('{out_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
    kept_rows, first, last = con.execute(f"""
        SELECT count(*), to_timestamp(min(ts))::DATE::VARCHAR,
               to_timestamp(max(ts))::DATE::VARCHAR
        FROM {final}
    """).fetchone()

    if not keep_staging:
        shutil.rmtree(staging, ignore_errors=True)

    return IngestStats(
        raw_rows=raw_rows,
        kept_rows=kept_rows,
        n_users=n_users,
        n_items=n_items,
        n_categories=n_cats,
        first_date=first,
        last_date=last,
        seconds=time.time() - t0,
        stage_seconds=stage_s,
    )
