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

---

# 2026-09-17 驗證紀錄

驗證日期：2026-09-17（Asia/Taipei）
基底 commit：`1f7115e`（推送前）

## 已通過

- `ruff check src tests scripts`：全部通過。
- `pytest -m "not network" -q`：**109 passed、2 deselected**，47.84 秒。
- 新增兩個 CLI 回歸測試，確認內容式通道不在預設通道中，且選用時必須明確指定
  與互動資料來自同一份映射的 `--items`，避免合成互動誤用正式商品表。
- 新增六個類別熱門通道測試，涵蓋類別內熱度、排除已見商品、多類別權重、
  cutoff 防洩漏、未知歷史與 fit 前呼叫。
- 新增兩個可及性診斷測試，確認 `channel_reachability` 寫入實驗紀錄；
  缺少 `category_idx` 時會留下明確的 `skipped` 原因。
- 全量 valid 權重掃描使用同一批 20,000 位使用者（樣本指紋 `4176b658…`），
  內容權重 0.15／0.25／0.35／0.50／1.00 均已跑完；Recall@500 在 0.25 最高。
- 類別熱門通道的 0.25／0.50／1.00 權重均已跑完；結果沒有超過內容式通道。

## 關鍵量測

- 內容式權重 0.25：Recall@10 = 0.00401、Recall@500 = 0.03023、
  NDCG@500 = 0.00703，為目前 valid 上最佳設定。
- 可及性診斷：60.3% 的答案落在使用者買過的類別內；只有 21.7% 的答案商品
  存在於共現圖中，代表共現通道無法觸及 78.3% 的答案。
- 類別熱門雖把候選聯集上限提高到 0.03936，最佳 Recall@500 只有 0.02995，
  因此不採用。
- ALS Top-K 向量化版本雖與原版逐位元等價，但效能降到 0.63x；
  只向量化已見商品排除在 176 個區塊、長歷史下也只有 0.79x，兩者均已還原。
- 在 0.72 GB、無法放入快取的商品因子矩陣上，`user_batch_size` 由 64 提高到
  1024 量到 1.48x；提高到 2048 反而降到 0.60x。這是縮小後的工程基準，
  不是 1.47 GB 全量模型的加速保證。

## 尚未驗證

- 尚未以鎖定設定執行最終 test 時間段；不能把 valid 結果稱為泛化或線上成效。
- 權重 0.15／0.25／0.35 之間尚未做 paired bootstrap，因此不宣稱 0.25
  顯著優於相鄰權重。
- 兩個標記為 network 的下載／續傳測試未執行。
- 尚未在 1.47 GB 的全量 ALS 商品因子矩陣上重跑大型 `user_batch_size` 基準。
