"""資料品質與分布報告。

這份報告的目的不是「看起來很完整」，而是回答幾個會直接改變設計的問題：

1. **合格使用者有多少？** next-item 預測需要使用者同時具備「切分點前的
   歷史」與「該段的新商品」。小樣本測試時 92.3 萬位使用者只產生 37 位
   合格者。若全量資料下仍然不足，切分方案就必須調整 —— 這是整個專案
   最大的單一風險。
2. **長尾有多長？** 決定 min_interactions 門檻該設在哪。
3. **時間分布如何？** 早年的資料量若極少，訓練視窗就該縮短。

報告同時輸出到主控台與 Markdown 檔，後者可直接放進專案的 reports/。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from amazon_recsys.evaluation import splits as S


@dataclass
class Report:
    """收集報告內容，最後一次輸出。"""

    lines: list[str] = field(default_factory=list)

    def h(self, title: str) -> None:
        self.lines.append(f"\n## {title}\n")

    def p(self, text: str) -> None:
        self.lines.append(text + "\n")

    def table(self, headers: list[str], rows: list[list]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def text(self) -> str:
        return "\n".join(self.lines)


def _fetch(con, sql: str) -> list[tuple]:
    return con.execute(sql).fetchall()


def build_report(con, src: str, split: S.TimeSplit = S.DEFAULT_SPLIT) -> Report:
    r = Report()
    r.lines.append("# 資料品質報告\n")

    # --- 整體規模 ---------------------------------------------------------
    n, nu, ni, nc, lo, hi = _fetch(con, f"""
        SELECT count(*), count(DISTINCT user_idx), count(DISTINCT item_idx),
               count(DISTINCT category_idx),
               to_timestamp(min(ts))::DATE::VARCHAR, to_timestamp(max(ts))::DATE::VARCHAR
        FROM {src}
    """)[0]
    r.h("整體規模")
    r.table(
        ["項目", "數值"],
        [["互動數", f"{n:,}"], ["使用者", f"{nu:,}"], ["商品", f"{ni:,}"],
         ["類別", nc], ["時間範圍", f"{lo} ~ {hi}"],
         ["平均每人互動數", f"{n / nu:.2f}"], ["平均每商品互動數", f"{n / ni:.2f}"]],
    )

    # --- 使用者互動分布（決定門檻的關鍵）----------------------------------
    rows = _fetch(con, f"""
        WITH uc AS (SELECT user_idx, count(*) AS n FROM {src} GROUP BY 1)
        SELECT
          count(*) FILTER (WHERE n = 1), count(*) FILTER (WHERE n BETWEEN 2 AND 4),
          count(*) FILTER (WHERE n BETWEEN 5 AND 9), count(*) FILTER (WHERE n BETWEEN 10 AND 49),
          count(*) FILTER (WHERE n >= 50), count(*), max(n),
          median(n), quantile_cont(n, 0.9), quantile_cont(n, 0.99)
        FROM uc
    """)[0]
    one, f2, f5, f10, f50, total_u, mx, med, p90, p99 = rows
    r.h("使用者互動分布")
    r.p("**這張表決定 min_interactions 門檻，也決定有多少使用者能參與評估。**")
    r.table(
        ["互動筆數", "使用者數", "佔比"],
        [[lab, f"{v:,}", f"{100 * v / total_u:.2f}%"]
         for lab, v in [("1 筆", one), ("2-4 筆", f2), ("5-9 筆", f5),
                        ("10-49 筆", f10), ("50 筆以上", f50)]],
    )
    r.p(f"中位數 {med:.0f}、P90 {p90:.0f}、P99 {p99:.0f}、最大 {mx:,}")

    # --- 商品長尾 ---------------------------------------------------------
    head_share, n_head = _fetch(con, f"""
        WITH ic AS (SELECT item_idx, count(*) AS n FROM {src} GROUP BY 1),
             ranked AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS rk,
                               count(*) OVER () AS total FROM ic)
        SELECT sum(n) FILTER (WHERE rk <= total * 0.01) * 100.0 / sum(n),
               count(*) FILTER (WHERE rk <= total * 0.01)
        FROM ranked
    """)[0]
    only_once = _fetch(con, f"""
        WITH ic AS (SELECT item_idx, count(*) AS n FROM {src} GROUP BY 1)
        SELECT count(*) FILTER (WHERE n = 1) * 100.0 / count(*) FROM ic
    """)[0][0]
    r.h("商品長尾程度")
    r.table(
        ["指標", "數值"],
        [["最熱門 1% 商品佔總互動比例", f"{head_share:.1f}%（{n_head:,} 個商品）"],
         ["只被互動過 1 次的商品佔比", f"{only_once:.1f}%"]],
    )

    # --- 時間分布 ---------------------------------------------------------
    yearly = _fetch(con, f"SELECT year, count(*) FROM {src} GROUP BY 1 ORDER BY 1 DESC LIMIT 12")
    r.h("近年互動量分布")
    r.table(["年份", "互動數", "佔總量"],
            [[y, f"{c:,}", f"{100 * c / n:.2f}%"] for y, c in yearly])

    # --- 時間切分與合格使用者（最關鍵的檢查）------------------------------
    r.h("時間切分與合格使用者")
    r.p(f"切分方案：{split.describe()}")
    r.table(["區段", "互動數", "使用者數", "商品數", "特徵可用到"],
            [[x["segment"], f"{x['interactions']:,}", f"{x['users']:,}",
              f"{x['items']:,}", x["cutoff"]]
             for x in S.split_summary(con, src, split)])

    for seg in ("valid", "test"):
        lo_b, hi_b = split.bounds(seg)
        cutoff = split.feature_cutoff(seg)
        eligible, with_new = _fetch(con, f"""
            WITH hist AS (SELECT DISTINCT user_idx FROM {src} WHERE ts < {cutoff}),
                 fut  AS (SELECT user_idx, count(DISTINCT item_idx) AS k
                          FROM {src} WHERE ts >= {lo_b} AND ts < {hi_b} GROUP BY 1)
            SELECT count(*), count(*) FILTER (WHERE k > 0)
            FROM fut JOIN hist USING (user_idx)
        """)[0]
        r.p(f"**{seg} 段**：同時具備歷史與該段互動的使用者 **{eligible:,}** 位"
            f"（其中 {with_new:,} 位有互動商品）")
    r.p("\n> 註：上述數字尚未扣除「該段只買了舊商品」的使用者，"
        "實際合格數會再少一些。真正的數字由 build_eval_set() 產生。")

    # --- 資料異常 ---------------------------------------------------------
    bad = _fetch(con, f"""
        SELECT count(*) FILTER (WHERE rating NOT BETWEEN 1 AND 5),
               count(*) FILTER (WHERE user_idx IS NULL OR item_idx IS NULL),
               count(*) FILTER (WHERE year(to_timestamp(ts)) <> year),
               round(100.0 * count(*) FILTER (WHERE verified_purchase) / count(*), 1),
               round(avg(rating), 3)
        FROM {src}
    """)[0]
    r.h("資料異常檢查")
    r.table(["檢查項目", "結果"],
            [["評分超出 1-5", f"{bad[0]:,}"], ["ID 為 NULL", f"{bad[1]:,}"],
             ["年分區與時間戳不符", f"{bad[2]:,}"],
             ["已驗證購買佔比", f"{bad[3]}%"], ["平均評分", bad[4]]])
    return r


def write_report(report: Report, path) -> None:
    buf = io.StringIO()
    buf.write(report.text())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
