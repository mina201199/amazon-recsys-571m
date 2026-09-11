"""平行下載 Amazon Reviews 2023 原始檔，支援斷點續傳。

實測（見 docs/specs/）：頻寬上限約 7 MB/s，8 條並行即飽和；
評論檔 71.2 GB 約需 2.9 小時，metadata 24.5 GB 約需 1 小時。

重跑本模組是安全的：已完成的檔案會跳過，未完成的從斷點接續，
不會重頭下載。
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from tqdm import tqdm

from amazon_recsys import config

# 大檔下載需要寬鬆的逾時；connect 短一點以便快速失敗重試
_TIMEOUT = httpx.Timeout(connect=30.0, read=120.0, write=120.0, pool=30.0)
_MAX_RETRIES = 4
_CHUNK = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class FileSpec:
    """一個待下載的檔案。"""

    url: str
    dest: Path

    @property
    def name(self) -> str:
        return self.dest.name


@dataclass
class Result:
    spec: FileSpec
    status: str  # "skipped" | "downloaded" | "failed"
    bytes_written: int
    remote_size: int | None = None
    error: str | None = None


def review_specs(categories: tuple[str, ...] = config.CATEGORIES) -> list[FileSpec]:
    """評論檔清單。"""
    return [
        FileSpec(
            url=f"{config.REVIEWS_BASE_URL}/{c}.jsonl.gz",
            dest=config.RAW_REVIEWS_DIR / f"{c}.jsonl.gz",
        )
        for c in categories
    ]


def meta_specs(categories: tuple[str, ...] = config.CATEGORIES) -> list[FileSpec]:
    """商品 metadata 檔清單。"""
    return [
        FileSpec(
            url=f"{config.META_BASE_URL}/meta_{c}.jsonl.gz",
            dest=config.RAW_META_DIR / f"meta_{c}.jsonl.gz",
        )
        for c in categories
    ]


def remote_size(client: httpx.Client, url: str) -> int | None:
    """用 HEAD 取得遠端檔案大小；取不到回傳 None。"""
    try:
        r = client.head(url, follow_redirects=True)
        r.raise_for_status()
        n = r.headers.get("content-length")
        return int(n) if n else None
    except Exception:
        return None


def _download_one(
    spec: FileSpec,
    client: httpx.Client,
    pbar: tqdm | None,
    lock: threading.Lock,
) -> Result:
    """下載單一檔案，必要時續傳。"""
    spec.dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(client, spec.url)
    local = spec.dest.stat().st_size if spec.dest.exists() else 0

    # 已完整下載 → 跳過
    if total is not None and local == total:
        if pbar is not None:
            with lock:
                pbar.update(total)
        return Result(spec, "skipped", 0, total)

    # 本機檔案比遠端還大 → 損毀，砍掉重來
    if total is not None and local > total:
        spec.dest.unlink()
        local = 0

    written = 0
    last_err: str | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            headers = {"Range": f"bytes={local}-"} if local else {}
            with client.stream(
                "GET", spec.url, headers=headers, follow_redirects=True
            ) as r:
                # 206 = 接受續傳；200 = 伺服器忽略 Range，必須從頭寫
                if local and r.status_code == 200:
                    local, mode = 0, "wb"
                elif r.status_code == 206:
                    mode = "ab"
                else:
                    r.raise_for_status()
                    mode = "wb"

                if pbar is not None and local:
                    with lock:
                        pbar.update(local)

                with open(spec.dest, mode) as f:
                    for chunk in r.iter_bytes(_CHUNK):
                        f.write(chunk)
                        written += len(chunk)
                        if pbar is not None:
                            with lock:
                                pbar.update(len(chunk))

            # 驗證最終大小
            final = spec.dest.stat().st_size
            if total is not None and final != total:
                raise OSError(f"大小不符：本機 {final} != 遠端 {total}")
            return Result(spec, "downloaded", written, total)

        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            local = spec.dest.stat().st_size if spec.dest.exists() else 0
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)  # 指數退避

    return Result(spec, "failed", written, total, last_err)


def download_all(
    specs: list[FileSpec],
    workers: int = config.DOWNLOAD_WORKERS,
    show_progress: bool = True,
) -> list[Result]:
    """平行下載一批檔案，回傳每個檔案的結果。"""
    limits = httpx.Limits(max_connections=workers + 4, max_keepalive_connections=workers)

    with httpx.Client(timeout=_TIMEOUT, limits=limits) as probe:
        sizes = {s.url: remote_size(probe, s.url) for s in specs}

    grand_total = sum(v for v in sizes.values() if v)
    already = sum(s.dest.stat().st_size for s in specs if s.dest.exists())
    print(
        f"共 {len(specs)} 個檔案，總計 {config.human_bytes(grand_total)}"
        f"（已有 {config.human_bytes(already)}，"
        f"待下載 {config.human_bytes(max(0, grand_total - already))}）"
    )

    lock = threading.Lock()
    pbar = (
        tqdm(total=grand_total, unit="B", unit_scale=True, unit_divisor=1024, desc="下載中")
        if show_progress
        else None
    )

    results: list[Result] = []
    with (
        httpx.Client(timeout=_TIMEOUT, limits=limits) as client,
        cf.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(_download_one, s, client, pbar, lock): s for s in specs}
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    if pbar is not None:
        pbar.close()
    return results


def summarise(results: list[Result]) -> None:
    """印出下載結果摘要。"""
    by_status: dict[str, list[Result]] = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    print("\n=== 下載摘要 ===")
    for status, label in (
        ("downloaded", "新下載"),
        ("skipped", "已存在，跳過"),
        ("failed", "失敗"),
    ):
        rs = by_status.get(status, [])
        if rs:
            total = sum(r.bytes_written for r in rs)
            print(f"  {label:14s} {len(rs):3d} 個檔案  {config.human_bytes(total)}")

    for r in by_status.get("failed", []):
        print(f"    ✗ {r.spec.name}: {r.error}")
    if by_status.get("failed"):
        print("\n  失敗的檔案重跑本腳本即可從斷點續傳。")
