"""專案路徑與資料集常數。

所有路徑集中在此，避免散落各處。程式碼在 C 槽的 git repo，
資料在 D 槽且永不進版控——兩者刻意分離。
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 路徑
# --------------------------------------------------------------------------
# 可用環境變數 AMAZON_DATA_ROOT 覆寫，換機器時不必改程式
DATA_ROOT = Path(os.environ.get("AMAZON_DATA_ROOT", r"D:\amazon-reviews-2023"))

RAW_DIR = DATA_ROOT / "raw"
RAW_REVIEWS_DIR = RAW_DIR / "reviews"
RAW_META_DIR = RAW_DIR / "meta"

PARQUET_DIR = DATA_ROOT / "parquet"
INTERACTIONS_DIR = PARQUET_DIR / "interactions"  # 核心互動表，year=YYYY 分區
ITEMS_DIR = PARQUET_DIR / "items"                # 商品屬性
MAPS_DIR = PARQUET_DIR / "maps"                  # user_id / item_id 整數映射

FEATURES_DIR = DATA_ROOT / "features"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

ALL_DIRS = (
    RAW_REVIEWS_DIR, RAW_META_DIR,
    INTERACTIONS_DIR, ITEMS_DIR, MAPS_DIR,
    FEATURES_DIR, REPORTS_DIR,
)


def ensure_dirs() -> None:
    """建立所有需要的目錄（冪等）。"""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 資料來源
# --------------------------------------------------------------------------
_BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
REVIEWS_BASE_URL = f"{_BASE}/review_categories"
META_BASE_URL = f"{_BASE}/meta_categories"

# 全部 34 個類別檔（33 個正式類別 + Unknown）。
# 依實測檔案大小由大到小排序：大檔先開始下載，讓 8 條連線的收尾更平均，
# 避免最後只剩一個大檔在跑、其他連線閒置。
CATEGORIES: tuple[str, ...] = (
    "Unknown",                      # 7.80 GB
    "Home_and_Kitchen",             # 7.74
    "Clothing_Shoes_and_Jewelry",   # 6.62
    "Electronics",                  # 6.03
    "Books",                        # 5.79
    "Kindle_Store",                 # 4.26
    "Tools_and_Home_Improvement",   # 3.27
    "Health_and_Household",         # 2.98
    "Beauty_and_Personal_Care",     # 2.76
    "Sports_and_Outdoors",          # 2.45
    "Cell_Phones_and_Accessories",  # 2.36
    "Movies_and_TV",                # 2.23
    "Pet_Supplies",                 # 2.17
    "Automotive",                   # 2.14
    "Patio_Lawn_and_Garden",        # 2.01
    "Toys_and_Games",               # 1.82
    "Grocery_and_Gourmet_Food",     # 1.52
    "Office_Products",              # 1.51
    "Arts_Crafts_and_Sewing",       # 0.96
    "CDs_and_Vinyl",                # 0.96
    "Baby_Products",                # 0.78
    "Video_Games",                  # 0.76
    "Industrial_and_Scientific",    # 0.63
    "Software",                     # 0.46
    "Musical_Instruments",          # 0.43
    "Amazon_Fashion",               # 0.27
    "Appliances",                   # 0.25
    "All_Beauty",                   # 0.09
    "Handmade_Products",            # 0.07
    "Health_and_Personal_Care",     # 0.06
    "Digital_Music",                # 0.02
    "Gift_Cards",                   # 0.01
    "Magazine_Subscriptions",       # 0.01
    "Subscription_Boxes",           # 0.004
)

# --------------------------------------------------------------------------
# 執行參數
# --------------------------------------------------------------------------
# 實測：1 條 3.87 MB/s、4 條 6.53、8 條 6.98 → 8 條已達頻寬飽和
DOWNLOAD_WORKERS = 8

# DuckDB 執行緒數（本機 i5-14500 為 20 執行緒）
N_THREADS = os.cpu_count() or 8


def human_bytes(n: float) -> str:
    """把位元組數轉成易讀字串。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"
