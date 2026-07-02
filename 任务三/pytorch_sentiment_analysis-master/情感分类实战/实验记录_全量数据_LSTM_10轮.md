# Sentiment140 全量實驗記錄（LSTM，10 輪）

## 1. 實驗目標

使用 `PyTorch` 實作的 `LSTM` 模型，在 Sentiment140 全量資料（160 萬筆）上進行情感二分類訓練，完整跑滿 10 個 epoch，並記錄訓練過程與最終指標。

## 2. 實驗環境

- 作業系統：Windows 10
- Python 環境：Conda `dl`
- 核心框架：`torch 2.7.1+cu118`
- CUDA：可用（`Torch CUDA available: True`）
- 裝置：`cuda`
- 主要依賴：`torchvision`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`tqdm`

## 3. 資料與切分

- 資料集：Sentiment140
- 載入樣本數：`1,600,000`
- 類別分布：`label 0 = 0.5`，`label 1 = 0.5`
- 切分方式：
  - Train：`1,280,000`
  - Validation：`160,000`
  - Test：`160,000`

## 4. 模型與訓練設定

### 4.1 模型結構

- 模型：`LSTMClassifier`
- 詞嵌入層：`Embedding(vocab_size=20000, embedding_dim=200, padding_idx=0)`
- 編碼器：單層 `LSTM(hidden_size=128, batch_first=True)`
- 正則化：`Dropout(0.2)`
- 輸出層：`Linear(128 -> 2)`（二分類）

### 4.2 超參數（本次實驗）

- `epochs=10`
- `batch_size=256`
- `embedding_dim=200`
- `hidden_size=128`
- `dropout=0.2`
- `lr=1e-3`
- `max_vocab_size=20000`
- `min_freq=2`
- `max_len=50`
- `val_size=0.1`
- `test_size=0.1`
- `seed=42`
- `max_samples=-1`（代表不抽樣，使用全量資料）

### 4.3 優化與損失

- 損失函數：`CrossEntropyLoss`
- 優化器：`Adam`
- 模型保存策略：以 `val_loss` 最佳為準，保存為 `best_lstm.pt`

## 5. 執行命令與輸出檔

### 5.1 實際執行命令

```powershell
conda run -n dl python "src/Pytorch_LSTM实战情感分类.py" --epochs 10 --max-samples -1
```

### 5.2 終端訓練可視化

程式已加入 `tqdm`，每個 epoch 的 `train/val/test` 都會顯示批次進度與即時 loss/acc，並在每輪結束後印出彙總行（`Epoch xx | ...`）。

### 5.3 主要輸出檔案

- 訓練日誌：`output/logs/train_full_e10_console.log`
- 每輪指標：`output/logs/training_history.csv`
- 最佳模型：`output/models/best_lstm.pt`
- 曲線圖：`output/figures/lstm_loss.png`、`output/figures/lstm_accuracy.png`

## 6. 訓練過程記錄（10 輪）

| Epoch | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4246 | 0.8032 | 0.3891 | 0.8234 | 0.3873 | 0.8249 |
| 2 | 0.3700 | 0.8338 | 0.3777 | 0.8301 | 0.3747 | 0.8321 |
| 3 | 0.3461 | 0.8464 | 0.3758 | 0.8317 | 0.3731 | 0.8332 |
| 4 | 0.3251 | 0.8571 | 0.3805 | 0.8315 | 0.3780 | 0.8329 |
| 5 | 0.3044 | 0.8676 | 0.3880 | 0.8314 | 0.3865 | 0.8319 |
| 6 | 0.2842 | 0.8778 | 0.3968 | 0.8291 | 0.3951 | 0.8309 |
| 7 | 0.2644 | 0.8874 | 0.4215 | 0.8268 | 0.4186 | 0.8275 |
| 8 | 0.2460 | 0.8963 | 0.4346 | 0.8254 | 0.4321 | 0.8270 |
| 9 | 0.2296 | 0.9044 | 0.4572 | 0.8218 | 0.4531 | 0.8239 |
| 10 | 0.2155 | 0.9108 | 0.4751 | 0.8210 | 0.4698 | 0.8228 |

## 7. 關鍵結果與分析

### 7.1 最佳 epoch（以驗證集 loss）

- 最佳 `val_loss`：`0.3758`（Epoch 3）
- 對應 `val_acc`：`0.8317`
- 對應 `test_acc`：`0.8332`

### 7.2 最終 epoch（Epoch 10）

- `train_acc=0.9108`
- `val_acc=0.8210`
- `test_acc=0.8228`

### 7.3 現象解讀

- 訓練集指標持續上升（`train_acc` 由 0.8032 -> 0.9108）。
- 驗證與測試在第 3 輪附近達到高點後，隨後緩慢下降。
- 呈現中後期過擬合跡象：模型持續記憶訓練資料，但泛化能力略下降。

## 8. 訓練耗時

- START_TIME：`2026-07-02 11:00:35`
- END_TIME：`2026-07-02 11:12:11`
- 總耗時：約 `11.61` 分鐘（GPU）

## 9. 結論

本次已在 `conda dl` 環境下完成 Sentiment140 全量資料訓練，10 輪實驗可重現且產物完整。模型在 Epoch 3 左右達到最佳泛化表現（`test_acc ≈ 0.8332`），後續訓練主要提升訓練集表現，泛化略有下滑。若追求最佳測試表現，建議採用早停（Early Stopping）或以最佳驗證 loss 的模型作為最終提交模型。

## 10. 後續優化建議（可選）

- 加入 `Early Stopping`（patience 2~3）
- 嘗試學習率衰減（如 `ReduceLROnPlateau`）
- 加強正則化（較大 dropout、weight decay）
- 增加文本預處理（去 URL/mention，保留否定詞模式）
- 嘗試雙向 LSTM 或 Transformer baseline 做對照
