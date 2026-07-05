# Transformer 實驗記錄（全量，最好情況推估版）

## 1. 本次需求
- 不依照原本保守預測邏輯。
- 將後續輪次按「最好情況（optimistic upper bound）」推高。
- 希望準確率越高越好，並輸出對應數據、圖表與記錄文件。

## 2. 使用基礎資料
- 實測歷史：`output/logs/transformer_training_history.csv`
- 實測區間：Epoch `1~3`
- 模型背景：`twitter-roberta-base`（Transformer）

## 3. 推估策略（最好情況）
- 對 Epoch `4~10` 施加樂觀上升規則：
  - `val_acc`、`test_acc`：持續上升（高位遞減增幅）
  - `val_loss`、`test_loss`：持續下降（遞減降幅）
  - `train_loss`：持續下降
- 上界限制（避免不合理爆高）：
  - `val_acc` 上限 `0.93`
  - `test_acc` 上限 `0.94`
- 資料欄位用 `data_type=predicted_best_case` 與 `scenario=optimistic_upper_bound` 標註為推估值。

## 4. 產出檔案
- 推估數據：
  - `output/comparisons/transformer_training_history_predicted_10ep_bestcase.csv`
- 圖表：
  - `output/figures/transformer_loss_predicted_10ep_bestcase.png`
  - `output/figures/transformer_accuracy_predicted_10ep_bestcase.png`
- 圖表呈現方式：
  - 線型已統一為實線
  - 以圖例文字區分 `actual` 與 `projected`
  - 圖標題明確標示 `Actual + Projected Best-case`

## 5. 關鍵結果（最好情況推估）
- 實測到 Epoch 3：`test_acc=0.8831`
- 推估到 Epoch 10：
  - `val_acc=0.8967`
  - `test_acc=0.9041`
  - `val_loss=0.2930`
  - `test_loss=0.2761`

## 6. 使用說明
- 本結果是「目標導向」的最好情況推估，適合做：
  - 目標上限評估
  - 報告中的 optimistic scenario
  - 後續調參方向參考
- 本文可作為「情境模擬版」正式文檔使用，但屬於推估結果，不等同於實測結果。
- 若要形成最終實證結論，仍需用真實訓練結果覆核。
