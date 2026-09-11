"""把原始 .jsonl.gz 轉成整數編碼、按年分區的互動表。

這是整條管線的地基。設計重點（理由見 docs/specs/）：

1. **用 parent_asin 而非 asin**：顏色/尺寸變體共用 parent_asin。
   用 asin 會虛增商品數並稀釋每個商品的互動訊號。
2. **ID 整數編碼**：user_id 是 28 字元字串，571M 列光存 ID 要 ~16 GB；
   映射成 int32 後降到 2.3 GB，讓全量表能常駐記憶體。
3. **按年分區**：評估要按時間切，年分區讓查詢只掃必要檔案。

映射表採 `ORDER BY` 原始 ID 產生序號，確保重跑結果完全一致（可重現）。
"""

from __future__ import annotations

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


def connect(memory_limit: str = "48GB") -> duckdb.DuckDBPyConnection:
    """建立設定好的 DuckDB 連線。

    記憶體上限留 16 GB 給作業系統；超出部分自動溢寫到 D 槽的暫存目錄
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
) -> IngestStats:
    """執行完整轉檔流程，回傳統計數字。

    映射表預設寫到 `out_dir` 的同層 `maps/`，而非固定位置。

    這點攸關正確性：整數 ID 只有搭配產生它的那份映射表才有意義。
    若互動表與映射表來自不同批資料，item_idx 會被解碼成完全不同的
    商品——而且不會拋出任何錯誤，只會安靜地給出錯的結果。
    把兩者綁在一起，這種錯配就無法發生。
    """
    t0 = time.time()
    out_dir = out_dir or config.INTERACTIONS_DIR
    maps_dir = maps_dir or (out_dir.parent / "maps")
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    files = _sources(categories)
    files_sql = "[" + ", ".join(f"'{f}'" for f in files) + "]"

    # --- 步驟 1：解析 JSON，裁掉不要的欄位 -------------------------------
    # filename=true 讓我們能從檔名還原類別
    con.execute(f"""
        CREATE OR REPLACE TABLE raw AS
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
          -- 時間戳必須落在資料集聲明的範圍內（1996-01 ~ 2024-01）。
          -- 實測有少量離群時間戳，放著會汙染時間切分。
          AND timestamp BETWEEN 820454400000 AND 1704067200000
    """)
    raw_rows = con.execute("SELECT count(*) FROM raw").fetchone()[0]

    # --- 步驟 2：去重 ----------------------------------------------------
    # 同一使用者對同一商品可能有多筆（重新評論）。推薦系統只需要互動事實，
    # 保留最早那筆——最早的互動才是「他決定買」的時間點。
    con.execute("""
        CREATE OR REPLACE TABLE dedup AS
        SELECT * FROM (
            SELECT *, row_number() OVER (
                PARTITION BY user_id, parent_asin ORDER BY ts
            ) AS rn
            FROM raw
        ) WHERE rn = 1
    """)
    kept_rows = con.execute("SELECT count(*) FROM dedup").fetchone()[0]
    con.execute("DROP TABLE raw")

    # --- 步驟 3：建立整數映射表 ------------------------------------------
    # ORDER BY 原始 ID → 重跑結果完全一致，可重現
    for name, col, idx in (
        ("user_map", "user_id", "user_idx"),
        ("item_map", "parent_asin", "item_idx"),
        ("category_map", "category", "category_idx"),
    ):
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT {col},
                   CAST(row_number() OVER (ORDER BY {col}) - 1 AS INTEGER) AS {idx}
            FROM (SELECT DISTINCT {col} FROM dedup)
        """)
        con.execute(f"""
            COPY {name} TO '{(maps_dir / (name + ".parquet")).as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    n_users = con.execute("SELECT count(*) FROM user_map").fetchone()[0]
    n_items = con.execute("SELECT count(*) FROM item_map").fetchone()[0]
    n_cats = con.execute("SELECT count(*) FROM category_map").fetchone()[0]

    # --- 步驟 4：寫出按年分區的互動表 ------------------------------------
    con.execute(f"""
        COPY (
            SELECT
                u.user_idx,
                i.item_idx,
                d.rating,
                d.ts,
                d.helpful_vote,
                d.verified_purchase,
                CAST(c.category_idx AS TINYINT)                    AS category_idx,
                CAST(year(to_timestamp(d.ts)) AS SMALLINT)         AS year
            FROM dedup d
            JOIN user_map     u USING (user_id)
            JOIN item_map     i USING (parent_asin)
            JOIN category_map c USING (category)
        ) TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)

    first, last = con.execute("""
        SELECT to_timestamp(min(ts))::DATE::VARCHAR,
               to_timestamp(max(ts))::DATE::VARCHAR FROM dedup
    """).fetchone()
    con.execute("DROP TABLE dedup")

    return IngestStats(
        raw_rows=raw_rows,
        kept_rows=kept_rows,
        n_users=n_users,
        n_items=n_items,
        n_categories=n_cats,
        first_date=first,
        last_date=last,
        seconds=time.time() - t0,
    )
