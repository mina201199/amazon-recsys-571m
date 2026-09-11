"""時間切分與資料洩漏防護。

推薦系統最常見的致命錯誤是隨機切分：用未來的互動預測過去，
分數會非常好看，但模型上線後毫無用處。本模組把「只能用切分點
之前的資料」這件事變成程式強制的約束，而非靠人自律。

三個邊界切出三段：

    ...........train_end.......valid_end.......test_end
    [    訓練     )[   驗證    )[   測試    )

評估某一段時，**只能**使用該段起點之前的資料當作歷史與特徵：

    評估驗證段 → 只能用 ts <  train_end
    評估測試段 → 只能用 ts <  valid_end

這條規則由 `feature_cutoff()` 提供，並由 `assert_no_leakage()` 驗證。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import duckdb

Segment = Literal["train", "valid", "test"]


def ts(date: str) -> int:
    """把 'YYYY-MM-DD' 轉成 UTC unix 秒數。"""
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp())


def to_date(unix_ts: int) -> str:
    return datetime.fromtimestamp(unix_ts, tz=UTC).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class TimeSplit:
    """三段式時間切分。邊界為左閉右開。"""

    train_end: int
    valid_end: int
    test_end: int

    def __post_init__(self) -> None:
        if not (self.train_end < self.valid_end < self.test_end):
            raise ValueError(
                f"邊界必須遞增：train_end={to_date(self.train_end)} "
                f"< valid_end={to_date(self.valid_end)} "
                f"< test_end={to_date(self.test_end)}"
            )

    def bounds(self, segment: Segment) -> tuple[int | None, int]:
        """該段的 [起點, 終點)。訓練段沒有起點（用上所有歷史）。"""
        return {
            "train": (None, self.train_end),
            "valid": (self.train_end, self.valid_end),
            "test": (self.valid_end, self.test_end),
        }[segment]

    def feature_cutoff(self, segment: Segment) -> int:
        """評估該段時，特徵與歷史只能取自這個時間點之前的資料。

        這是整個模組的核心：所有特徵計算都必須以此為上限。
        """
        lo, hi = self.bounds(segment)
        return hi if lo is None else lo

    def describe(self) -> str:
        return (
            f"訓練 ~{to_date(self.train_end)} | "
            f"驗證 {to_date(self.train_end)}~{to_date(self.valid_end)} | "
            f"測試 {to_date(self.valid_end)}~{to_date(self.test_end)}"
        )


# 預設切分（見 docs/specs/ §7.1）。資料集止於 2023-09-09，
# 測試段終點設為 09-10 以完整涵蓋最後一天。
DEFAULT_SPLIT = TimeSplit(
    train_end=ts("2023-03-01"),
    valid_end=ts("2023-06-01"),
    test_end=ts("2023-09-10"),
)


def split_summary(
    con: duckdb.DuckDBPyConnection, src: str, split: TimeSplit = DEFAULT_SPLIT
) -> list[dict]:
    """各段的互動數、使用者數、商品數。"""
    rows = []
    for seg in ("train", "valid", "test"):
        lo, hi = split.bounds(seg)
        where = f"ts < {hi}" if lo is None else f"ts >= {lo} AND ts < {hi}"
        n, nu, ni = con.execute(
            f"SELECT count(*), count(DISTINCT user_idx), count(DISTINCT item_idx) "
            f"FROM {src} WHERE {where}"
        ).fetchone()
        rows.append(
            {"segment": seg, "interactions": n, "users": nu, "items": ni,
             "cutoff": to_date(split.feature_cutoff(seg))}
        )
    return rows


def build_eval_set(
    con: duckdb.DuckDBPyConnection,
    src: str,
    split: TimeSplit = DEFAULT_SPLIT,
    segment: Segment = "test",
    min_history: int = 1,
    max_users: int | None = None,
    seed: int = 42,
) -> tuple[list[int], list[list[int]], list[set[int]]]:
    """建立評估資料：每位使用者的歷史，以及他在該段真正買了什麼。

    兩個刻意的設計決定：

    1. **歷史取自 feature_cutoff 之前**，不是該段之前的任意時間 ——
       這讓洩漏在資料建構階段就不可能發生。
    2. **正確答案排除使用者已經互動過的商品** —— 推薦一個他早就買過的
       東西不算本事，留著會讓分數虛高。

    回傳 (user_idx 清單, 歷史清單, 正確答案集合清單)。
    """
    lo, hi = split.bounds(segment)
    if lo is None:
        raise ValueError("訓練段不能當評估目標")
    cutoff = split.feature_cutoff(segment)

    # 先決定要哪些使用者，再只為他們建立歷史陣列。
    #
    # 順序很重要：反過來寫（先為全部使用者建歷史、最後才抽樣）會對
    # 5367 萬位使用者、5.5 億筆互動做 list 聚合，然後丟掉 99.9%。
    # 先篩後建，工作量少好幾個數量級。
    sample = ""
    if max_users is not None:
        # hash 抽樣：同一 seed 下結果穩定，且不需要排序全表
        sample = f"AND hash(user_idx * 2654435761 + {seed}) % 1000000 < {max_users}"

    rows = con.execute(f"""
        WITH eligible AS (
            -- 該段有互動、且抽樣命中的使用者
            SELECT DISTINCT user_idx FROM {src}
            WHERE ts >= {lo} AND ts < {hi} {sample}
        ),
        hist AS (
            SELECT user_idx, list(item_idx ORDER BY ts) AS history
            FROM {src} WHERE ts < {cutoff}
              AND user_idx IN (SELECT user_idx FROM eligible)
            GROUP BY user_idx
        ),
        fut AS (
            SELECT user_idx, list(DISTINCT item_idx) AS future
            FROM {src} WHERE ts >= {lo} AND ts < {hi}
              AND user_idx IN (SELECT user_idx FROM eligible)
            GROUP BY user_idx
        )
        SELECT h.user_idx, h.history,
               -- 排除已互動過的商品
               list_filter(f.future, x -> NOT list_contains(h.history, x)) AS truth
        FROM hist h JOIN fut f USING (user_idx)
        WHERE length(h.history) >= {min_history}
    """).fetchall()

    users, histories, truths = [], [], []
    for uid, hist, truth in rows:
        if not truth:  # 買的全是舊商品 → 對本任務沒有評估價值
            continue
        users.append(uid)
        histories.append(list(hist))
        truths.append(set(truth))
    return users, histories, truths


def assert_no_leakage(
    con: duckdb.DuckDBPyConnection,
    table: str,
    cutoff: int,
    ts_col: str = "ts",
) -> None:
    """確認某張特徵表不含切分點之後的資料。

    這是在每個特徵產出後都該呼叫的守門員。靠人記得「不要用未來資料」
    是不可靠的；讓程式在違規時直接爆炸才可靠。
    """
    latest = con.execute(f"SELECT max({ts_col}) FROM {table}").fetchone()[0]
    if latest is not None and latest >= cutoff:
        raise AssertionError(
            f"資料洩漏：{table} 含有 {to_date(latest)} 的資料，"
            f"但切分點是 {to_date(cutoff)}"
        )
