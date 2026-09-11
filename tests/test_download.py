"""下載器測試。

續傳是這支程式最關鍵的行為：全量下載約 2.9 小時，中斷後若拼接錯誤，
要到幾小時後解析 GB 級檔案時才會發現。因此必須用雜湊驗證，
而不只是檢查檔案大小。

需要網路的測試用 Subscription_Boxes（2.6 MB，全資料集最小的檔案）。
"""

from __future__ import annotations

import hashlib
import os

import pytest

from amazon_recsys import config
from amazon_recsys.ingest import download

SMALL = "Subscription_Boxes"  # 2.6 MB


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_review_specs_cover_all_categories():
    specs = download.review_specs()
    assert len(specs) == len(config.CATEGORIES)
    assert all(s.dest.name.endswith(".jsonl.gz") for s in specs)
    assert all(s.url.startswith(config.REVIEWS_BASE_URL) for s in specs)


def test_meta_specs_are_prefixed():
    specs = download.meta_specs((SMALL,))
    assert specs[0].dest.name == f"meta_{SMALL}.jsonl.gz"


@pytest.mark.network
def test_resume_produces_identical_file(tmp_path):
    """截斷後續傳，結果必須與完整下載 byte-for-byte 相同。"""
    spec = download.FileSpec(
        url=f"{config.REVIEWS_BASE_URL}/{SMALL}.jsonl.gz",
        dest=tmp_path / f"{SMALL}.jsonl.gz",
    )

    # 完整下載一次當基準
    (r1,) = download.download_all([spec], workers=1, show_progress=False)
    assert r1.status == "downloaded"
    full_hash, full_size = _sha256(spec.dest), spec.dest.stat().st_size

    # 截斷至 40%，模擬中斷
    os.truncate(spec.dest, full_size * 40 // 100)

    # 重跑：應只補下載缺少的部分
    (r2,) = download.download_all([spec], workers=1, show_progress=False)
    assert r2.status == "downloaded"
    assert r2.bytes_written < full_size, "應為續傳，而非重新下載整個檔案"
    assert spec.dest.stat().st_size == full_size
    assert _sha256(spec.dest) == full_hash, "續傳後內容與完整下載不符"


@pytest.mark.network
def test_completed_file_is_skipped(tmp_path):
    """已完成的檔案必須跳過，不可重新下載。"""
    spec = download.FileSpec(
        url=f"{config.REVIEWS_BASE_URL}/{SMALL}.jsonl.gz",
        dest=tmp_path / f"{SMALL}.jsonl.gz",
    )
    download.download_all([spec], workers=1, show_progress=False)
    (r,) = download.download_all([spec], workers=1, show_progress=False)
    assert r.status == "skipped"
    assert r.bytes_written == 0
