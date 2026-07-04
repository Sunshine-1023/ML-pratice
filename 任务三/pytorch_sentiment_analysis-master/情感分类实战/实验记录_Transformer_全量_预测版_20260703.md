# Transformer 實驗記錄（全量資料，預測版）

## 1. 目的
- 基於現有 Transformer 訓練結果，產出一份「可視化預測數據」與圖表。
- 不重新啟動訓練任務，直接用現有歷史紀錄推估後續輪次表現。

## 2. 模型與資料背景
- 模型：`cardiffnlp/twitter-roberta-base`
- 任務：Sentiment140 二分類
- 資料：全量資料流程（訓練歷史來源為既有 `transformer_training_history.csv`）

## 3. 實測資料來源
- 原始訓練歷史：
  - `output/logs/transformer_training_history.csv`
- 已有實測輪次：
  - Epoch `1~3`（`data_type=actual`）

## 4. 預測生成方式（擬真推估）
- 目標：補齊到 Epoch `10` 的趨勢數據（Epoch `4~10`）。
- 方法：依據前 3 輪趨勢做保守外推，並加入遞減步長，模擬常見微調後段行為：
  - `train_loss`：持續下降，但降幅逐步收斂
  - `val_loss` / `test_loss`：緩慢上升（模擬過擬合風險）
  - `val_acc` / `test_acc`：高位小幅回落（模擬平台期）
- 注意：本文件中的 Epoch `4~10` 為**預測值**，不是實測值。

## 5. 產出檔案
- 預測數據 CSV：
  - `output/comparisons/transformer_training_history_predicted_10ep.csv`
- 圖表：
  - `output/figures/transformer_loss_predicted_10ep.png`
  - `output/figures/transformer_accuracy_predicted_10ep.png`

## 6. 預測結果摘要
- 實測（Epoch 1~3）：
  - `test_acc` 由 `0.8808` 提升到 `0.8831`
- 預測（Epoch 4~10）：
  - `train_loss` 持續下降到約 `0.0573`
  - `val_loss` 緩升到約 `0.4360`
  - `test_acc` 在高位緩降到約 `0.8817`

## 7. 實測與預測分界
- `data_type=actual`：真實訓練輸出（可直接用於結論）
- `data_type=predicted`：根據趨勢外推（僅供規劃與預判）

## 8. 解讀建議
- 若你要最終報告或提交結果，建議以 `actual` 為主。
- `predicted` 可用來：
  - 預估是否值得繼續加 epoch
  - 觀察可能過擬合區段
  - 設計早停與學習率衰減策略
- 若要得到嚴格結論，建議後續跑完整 10 輪真實訓練再對照本預測。
