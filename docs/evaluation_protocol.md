# 評估協定與實驗紀錄

## 任務

輸入是 cutoff 前的評論互動，答案是未來時間窗內、歷史中未出現的商品集合。
評論不是完整購買資料；目前不按 rating 或 verified_purchase 過濾。

- valid：歷史 `ts < 2023-03-01`；答案 `[2023-03-01, 2023-06-01)`。
- test：歷史 `ts < 2023-06-01`；答案 `[2023-06-01, 2023-09-10)`。
- 品質報告到 09-14；09-10 起的資料不在固定測試協定內，勿稱涵蓋完整末日。
- 使用者至少一筆歷史、至少一個未見答案。排除沒有歷史的使用者，應揭露此選樣限制。
- `--max-users N --seed S` 在合格使用者中依 hash 排序取至多 N 人；若合格人數足夠，恰為 N。
  最後依 user_idx 排序。固定資料與 DuckDB 版本時可重現；跨資料版本須對照樣本指紋。

## 指標

Recall@K、NDCG@K 都先對每位使用者計算，再取 macro average。
coverage 使用 cutoff 前的 distinct 商品數作分母；不是整個資料集的商品目錄數。
所有通道保留同一批使用者與答案，包括模型不認識的歷史／答案。

`cold_truth_fraction_micro` 是所有答案互動中，cutoff 前未出現商品的占比。
`train_catalogue_recall_ceiling_macro` 是各使用者答案中可由訓練目錄觸及的比例平均。
`union_recall_ceiling` 是合併截斷前所有通道候選聯集的答案覆蓋；不等於受 K 限制後的成績。

paired bootstrap 對同一批使用者的 Recall 差值重抽樣，預設 1,000 次、95% percentile CI。
CI 衡量抽樣變動，未校正調參與多重比較，不能替代最終留出的 test 或線上 A/B 測試。
少量合成資料的 CI 只用來驗證程式，不做推論。

## 新實驗輸出

06_recall_eval.py 每個通道完成即儲存 JSON，失敗時記下 status=failed。
指定的既有結果檔不覆寫；預設檔名含 UTC 時間。

紀錄包括：

- 路徑值遮蔽後的 argv、repo 相對或 external 標示的 source、git commit / dirty status、
  程式內容 SHA256。公開紀錄不保存 Windows 使用者目錄。
- Python / 套件版本、資料檔路徑 / 大小 / 修改時間清單及清單 SHA256。
- 資料清單指紋不是資料內容雜湊；若需不可變資料證據，另外封存來源 manifest 與內容 SHA256。
- 時間切分、seed、實際使用者數、使用者順序指紋、平均歷史與答案長度。
- 每通道 dataclass 參數、模型規模、fit / recommend 秒數、融合權重與結果。
- 冷商品比例、候選聯集上限、歷史長度分群、paired bootstrap。

預設不記錄完整 raw review、真實 user_id 或每人推薦清單。
可加 `--examples 3` 保存固定排序前三位的整數 ID 歷史、答案與推薦；不挑命中案例。
商品名稱需要原資料對應的映射表與 metadata，不能用合成名稱代替。
單位是離線批次時間；不能稱為線上 P95 latency。DuckDB memory_limit 不限制 NumPy / ALS 全部記憶體。

## 原機重跑順序

PowerShell：

```powershell
$env:PYTHONUTF8 = '1'
$env:AMAZON_DATA_ROOT = 'D:\amazon-reviews-2023'
uv sync --frozen --extra dev
uv run pytest -m 'not network' -ra
uv run python scripts/06_recall_eval.py --segment valid --max-users 20000 --seed 42 --channels popularity covisitation --weights 1 1 --output reports/runs/valid-two-channel.json
uv run python scripts/06_recall_eval.py --segment valid --max-users 20000 --seed 42 --channels popularity covisitation als --weights 1 1 1 --output reports/runs/valid-three-channel.json
uv run python scripts/06_recall_eval.py --segment valid --max-users 20000 --seed 42 --channels popularity covisitation content --weights 1 1 0.25 --items D:/amazon-reviews-2023/parquet/items/items.parquet --output reports/runs/valid-content-w025.json
```

`--channels` 預設為 `popularity covisitation als`；content 不在預設清單，
必須明確指定並搭配 `--items`。屬性表與 `--src` 必須出自同一份 `item_map`——
先前 items_table 綁死在全域設定路徑，用合成展示資料評估時會靜靜載入正式目錄的
商品表，item_idx 指向完全不同的商品且不報錯。現已移除該回退路徑。

每次評估同時輸出 round-robin 與 RRF，避免替換融合方式後遺失對照。
可在 valid 改 weights（例如 1 / 0.5 / 0.25）做預先規劃的小型比較，這些值只是候選，並未優化。
選定設定後記錄理由與檔名，鎖定設定，再以 `--segment test` 執行一次。
valid 上已比較等權 1／1／1 與 1／1／0.25（內容式通道），0.25 在 Recall@10 與
Recall@500 都較佳且保住候選聯集上限，暫定為選用設定。權重 0.5 與 0.15 的兩次
執行沒有跑完，`status` 仍是 `running`，因此這不是完整掃描。test 尚未執行。

舊的 21,317 人結果缺少原始命令；禁止補寫推測的 max-users、seed 或 CI。
要比較新舊 ALS，可另外保存舊 commit 的結果與新結果，使用完全相同的新抽樣協定；
不能直接把 README 舊表和新抽樣結果相減，宣稱是演算法帶來的提升。
