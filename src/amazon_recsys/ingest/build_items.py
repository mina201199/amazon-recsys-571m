"""把商品 metadata 轉成整數編碼的屬性表。

內容式召回需要商品的靜態屬性——這是唯一能碰到冷啟動商品的資訊來源：
一件從未被買過的商品沒有共現紀錄、沒有熱度、也沒有潛在向量，
但它有類別、品牌與價格。

## 哪些欄位不能用（洩漏）

metadata 中的 `average_rating` 與 `rating_number` 是**資料集收集當下
（2023-09）的彙總值**，包含切分點之後產生的評論。拿來排序等於偷看未來，
而且不會報錯——只會讓離線分數虛高。

因此本模組**只保留不隨時間變動的靜態屬性**：

    main_category  商品所屬類別
    store          品牌／賣家
    price          標價（快照值，見下方註記）

品質與熱度訊號一律由互動表在切分點前重新計算，不取自 metadata。

`price` 嚴格說也是快照，但它是商品屬性而非結果變數，且不隨使用者行為
變動；本專案將其視為可用，並在報告中揭露這個取捨。

## 為什麼要 join item_map

metadata 以 `parent_asin` 為鍵，而互動表已整數編碼。兩者必須用
**產生互動表的那份 item_map** 對應，否則 item_idx 會指向錯誤的商品。
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

from amazon_recsys import config


@dataclass
class ItemStats:
    raw_rows: int
    matched_rows: int
    unmatched_rows: int
    with_price: int
    with_store: int
    n_categories: int
    n_stores: int
    seconds: float

    @property
    def match_rate(self) -> float:
        return self.matched_rows / max(self.raw_rows, 1)


def _sources(categories: tuple[str, ...] | None) -> list[str]:
    cats = categories or config.CATEGORIES
    paths = [config.RAW_META_DIR / f"meta_{c}.jsonl.gz" for c in cats]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 {len(missing)} 個 metadata 檔案：{missing[:5]}"
            "（請先執行 scripts/01_download.py --what meta）"
        )
    return [p.as_posix() for p in paths]


def build(
    con: duckdb.DuckDBPyConnection,
    categories: tuple[str, ...] | None = None,
    out_dir: Path | None = None,
    maps_dir: Path | None = None,
) -> ItemStats:
    """讀取 metadata，對應 item_idx，輸出屬性表。"""
    t0 = time.time()
    out_dir = out_dir or config.ITEMS_DIR
    maps_dir = maps_dir or (out_dir.parent / "maps")
    item_map = maps_dir / "item_map.parquet"
    if not item_map.exists():
        raise FileNotFoundError(
            f"找不到 {item_map}——屬性表必須對應產生互動表的那份映射，"
            "否則 item_idx 會指向錯誤的商品。"
        )

    files_sql = "[" + ", ".join(f"'{f}'" for f in _sources(categories)) + "]"
    staging = out_dir.parent / "_items_staging"
    shutil.rmtree(staging, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 階段 1：串流解析，只取靜態欄位。
    # price 在原始資料中是字串（如 "$12.99"、"None"、空字串），先清洗再轉型。
    con.execute(f"""
        COPY (
            SELECT
                parent_asin,
                main_category,
                store,
                try_cast(
                    nullif(replace(replace(CAST(price AS VARCHAR), '$', ''), ',', ''), 'None')
                    AS DOUBLE
                ) AS price
            FROM read_json_auto({files_sql}, ignore_errors=true,
                                union_by_name=true, maximum_object_size=20000000)
            WHERE parent_asin IS NOT NULL
        ) TO '{staging.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PER_THREAD_OUTPUT true, OVERWRITE_OR_IGNORE)
    """)
    src = f"read_parquet('{staging.as_posix()}/*.parquet')"
    raw_rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    # 階段 2：類別與品牌各自編碼成整數。
    # 品牌有數百萬個不重複值，存字串會讓屬性表膨脹數倍。
    for name, col, idx in (
        ("meta_category_map", "main_category", "main_category_idx"),
        ("store_map", "store", "store_idx"),
    ):
        con.execute(f"""
            CREATE OR REPLACE TABLE {name} AS
            SELECT {col}, CAST(row_number() OVER (ORDER BY {col}) - 1 AS INTEGER) AS {idx}
            FROM (SELECT DISTINCT {col} FROM {src} WHERE {col} IS NOT NULL)
        """)
        con.execute(f"""COPY {name} TO
            '{(maps_dir / (name + ".parquet")).as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)""")

    # 階段 3：對應 item_idx 並輸出。
    # 內連接：沒有任何互動的商品不在 item_map 中，也就無從推薦。
    con.execute(f"""
        COPY (
            SELECT
                i.item_idx,
                CAST(c.main_category_idx AS SMALLINT) AS main_category_idx,
                s.store_idx,
                m.price
            FROM {src} m
            JOIN read_parquet('{item_map.as_posix()}') i USING (parent_asin)
            LEFT JOIN meta_category_map c ON c.main_category = m.main_category
            LEFT JOIN store_map          s ON s.store         = m.store
        ) TO '{(out_dir / "items.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    out = f"read_parquet('{(out_dir / 'items.parquet').as_posix()}')"
    matched, with_price, with_store, n_cat, n_store = con.execute(f"""
        SELECT count(*),
               count(*) FILTER (WHERE price IS NOT NULL),
               count(*) FILTER (WHERE store_idx IS NOT NULL),
               count(DISTINCT main_category_idx),
               count(DISTINCT store_idx)
        FROM {out}
    """).fetchone()

    shutil.rmtree(staging, ignore_errors=True)
    return ItemStats(
        raw_rows=raw_rows,
        matched_rows=matched,
        unmatched_rows=raw_rows - matched,
        with_price=with_price,
        with_store=with_store,
        n_categories=n_cat,
        n_stores=n_store,
        seconds=time.time() - t0,
    )
