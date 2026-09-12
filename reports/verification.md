# 本次驗證紀錄

驗證日期：2026-09-12（Asia/Taipei）  
基底 commit：`c5e2cae9e56128b1d8b9a5c1a7886df5505e7613`

## 已通過

- `uv sync --frozen --extra dev`：依 `uv.lock` 建立 Python 3.12.14 環境。
- `ruff check src tests scripts`：全部通過。
- `pytest -m "not network" -ra`：**82 passed、8 skipped、2 deselected**，41.50 秒。
- 8 個 skip 是原有 `test_build_interactions.py` 需要兩個已下載原始檔；本次環境沒有 D 槽資料。
- 2 個 deselected 是標記為 network 的真實下載／續傳測試。
- 新增的 `test_ingest_synthetic.py` 以合成 gzip JSON 實際跑過解析、映射、去重及 Parquet 輸出，
  因此資料管線仍有一條不依賴外部資料的端到端測試。
- `scripts/07_demo.py` 成功產生 `reports/demo/demo.html`、`demo.json` 與合成 Parquet。
- `scripts/06_recall_eval.py` 成功在合成 Parquet 上跑完三通道、兩種融合、bootstrap、
  冷商品診斷與三位範例輸出，紀錄在 `reports/runs/demo-eval.json`。

## 本次測試涵蓋的關鍵修正

- ALS fold-in 對照手算正規方程。
- 商品分塊 Top-K 對照完整精確搜尋，含跨區塊同分排序。
- 商品數增加時仍能保留多使用者批次。
- 合格使用者中 hash top-N 的人數上限與 seed 可重現性。
- 加權 RRF 的共識、權重、重複、padding 與參數驗證。
- paired Recall bootstrap 的已知差值與 seed 重現。
- cutoff 前商品目錄、冷答案比例及理論可達上限。
- 評估 CLI 的成功紀錄、失敗紀錄與拒絕覆寫。
- 公開實驗 JSON 的本機絕對路徑遮蔽。

## 未驗證

- 沒有 71.2 GB 原始資料與 6 GB 全量 Parquet，因此未重跑全量資料品質、模型成績或耗時。
- 舊歷史表沒有原始命令、樣本 ID 或逐使用者結果，所以沒有替它補算信賴區間。
- 未執行兩個 network 測試。
- Codex 瀏覽器安全政策拒絕開啟本機 `file://` 頁面；沒有繞過政策做瀏覽器視覺 QA。
  已由產生器與後續結構／連結檢查驗證檔案內容。面試前請在一般瀏覽器開啟一次確認字型與視窗尺寸。

合成展示的指標只證明流程有輸出，不代表 Amazon 全量模型品質。
