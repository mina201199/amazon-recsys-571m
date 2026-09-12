"""不需下載資料的三位使用者展示；所有商品與互動均為合成資料。

產生 demo.html、demo.json、interactions/*.parquet。這些結果只驗證流程，
不能當作 Amazon 資料集的推薦品質證據。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb

from amazon_recsys.evaluation import metrics as M
from amazon_recsys.evaluation import splits as S
from amazon_recsys.recall import base
from amazon_recsys.recall.als import ALSRecall
from amazon_recsys.recall.covisitation import CoVisitationRecall
from amazon_recsys.recall.popularity import PopularityRecall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("reports/demo"))
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"輸出已存在，請指定新目錄：{args.out}")
    args.out.mkdir(parents=True)
    source = args.out / "interactions"
    source.mkdir()
    recent = S.ts("2023-02-01")
    future = S.ts("2023-04-01")
    rows = []
    # 40 位群組 A、20 位群組 B，各有 12 筆循環位移的歷史。
    for user in range(60):
        offset = 1 if user < 40 else 101
        rows.extend((user, offset + (user + j) % 20, recent) for j in range(12))
    histories = [[1, 2, 3], [101, 102, 103], [9999]]
    truths = [{4, 5}, {104}, {205}]
    for user, (hist, truth) in enumerate(zip(histories, truths, strict=True), start=1000):
        rows.extend((user, item, recent) for item in hist)
        rows.extend((user, item, future) for item in sorted(truth))
    names = {i: f"示意書籍 {i:02d}" for i in range(1, 21)}
    names.update({i: f"示意戶外用品 {i - 100:02d}" for i in range(101, 121)})
    names.update({9999: "示意長尾商品（僅一次互動）", 205: "示意新商品（切分後才出現）"})
    k = 10
    with duckdb.connect() as con:
        con.execute("CREATE TABLE inter(user_idx INT, item_idx INT, ts INT)")
        con.executemany("INSERT INTO inter VALUES (?, ?, ?)", rows)
        dest = (source / "demo.parquet").as_posix().replace("'", "''")
        con.execute(f"COPY inter TO '{dest}' (FORMAT PARQUET)")
        channels = [PopularityRecall(), CoVisitationRecall(
            window_days=365, max_items_per_user=20, top_n_neighbours=50, min_cooccurrence=3),
            ALSRecall(factors=32, iterations=10,
                      min_user_interactions=10, min_item_interactions=5)]
        recs = {}
        for channel in channels:
            channel.fit(con, "inter", S.DEFAULT_SPLIT.train_end)
            recs[channel.name] = channel.recommend(histories, k)
        candidates = list(recs.values())
        recs["round_robin"] = base.merge_channels(candidates, k)
        recs["weighted_rrf"] = base.weighted_rrf(candidates, k, [1.0, 1.0, 1.0])
    notes = [
        "觀察個人歷史如何影響候選；綠色代表命中本例未來答案。",
        "熱門榜可能偏向人數較多的群組 A；對照共現與 ALS 是否找回群組 B。",
        "歷史商品未通過 ALS 門檻，未來商品又在切分後才出現。此例呈現冷啟動限制，不能保證命中。",
    ]
    cases = []
    cards = []
    for u, hist in enumerate(histories):
        case = {"synthetic_user": 1000 + u, "history": hist, "truth": sorted(truths[u]),
                "recommendations": {name: row[u].tolist() for name, row in recs.items()}}
        cases.append(case)
        sections = []
        for name, matrix in recs.items():
            items = []
            for item in matrix[u]:
                if item < 0:
                    continue
                hit = int(item) in truths[u]
                label = html.escape(names[int(item)])
                items.append(f'<li class="{"hit" if hit else ""}">{label}'
                             f'<small>#{item}{" · 命中" if hit else ""}</small></li>')
            listing = (
                "<ol>" + "".join(items) + "</ol>"
                if items
                else "<p>沒有候選，交由熱門通道補足。</p>"
            )
            sections.append(f'<details {"open" if name == "weighted_rrf" else ""}>'
                            f"<summary>{html.escape(name)} · {len(items)} 個候選</summary>"
                            f"{listing}</details>")
        history_text = '、'.join(names[i] for i in hist)
        truth_text = '、'.join(names[i] for i in sorted(truths[u]))
        cards.append(f'<article><p class="eyebrow">CASE {u + 1:02d}</p>'
                     f'<h2>使用者 {1000 + u}</h2><p>{notes[u]}</p>'
                     f'<p><b>歷史：</b>{history_text}</p><p><b>未來答案：</b>{truth_text}</p>'
                     + ''.join(sections) + '</article>')
    report = {
        "data_kind": "SYNTHETIC_ONLY", "purpose": "workflow demonstration; not Amazon evaluation",
        "n_interactions": len(rows), "feature_cutoff": "2023-03-01 exclusive",
        "rrf_weights": [1, 1, 1], "weights_tuned": False, "item_names": names,
        "cases": cases,
        "metrics": {name: M.evaluate(matrix, truths, ks=(10,)).metrics
                    for name, matrix in recs.items()},
    }
    (args.out / "demo.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>三位使用者的召回展示</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f4f5f7;color:#192838;
font:16px/1.7 system-ui,"Microsoft JhengHei",sans-serif}
main{max-width:1100px;margin:auto;padding:48px 24px}
h1{font-size:36px;line-height:1.2;letter-spacing:-1px}
header{max-width:780px;margin-bottom:32px}
.notice{background:#fff1d4;border-left:4px solid #bd741b;padding:16px 20px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}
article{padding:24px;background:white;border:1px solid #d9e0e6;border-radius:12px}
h2{font-size:22px;margin-top:0}
.eyebrow{letter-spacing:2px;color:#536c85;font-size:12px;font-weight:bold}
details{border-top:1px solid #e1e6eb;padding:12px 0}
summary{cursor:pointer;font-weight:600;font-size:14px}
ol{padding-left:22px}li{padding:5px 3px;font-size:14px}
small{display:block;color:#5b6875}
.hit{color:#006a4e;background:#e4f7ee;border-radius:4px}
footer{margin-top:32px;color:#526171}
@media(max-width:850px){.grid{grid-template-columns:1fr}h1{font-size:30px}}
</style><main><header><p class="eyebrow">AMAZON RECSYS · OFFLINE DEMO</p>
<h1>從歷史互動，到推薦候選</h1>
<p>展開各通道，比較熱門、共現、ALS 與兩種合併方式的實際輸出。</p>
<p class="notice"><b>這是合成資料展示。</b>所有商品名稱與使用者均為示意；
由真實程式運算產出，不代表 Amazon 全量成績。等權 RRF 尚未調參，
也不保證優於基線。</p>
<p>模型只能看到 2023-03-01 之前的資料。綠色標示命中未來答案；
未命中同樣保留，方便討論限制。</p></header><section class="grid">'''
    page += "".join(cards)
    page += """</section><footer>目前完成離線候選生成與評估。
精排、線上服務與 A/B 測試尚未完成。</footer></main></html>"""
    (args.out / "demo.html").write_text(page, encoding="utf-8")
    print(f"展示已產生：{args.out / 'demo.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
