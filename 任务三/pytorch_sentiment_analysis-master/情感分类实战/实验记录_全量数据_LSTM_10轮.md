# Sentiment140 全量實驗記錄（LSTM，10 輪）

> 實驗日期：2026-07-02  
> 執行環境：Conda `dl`（CUDA GPU）  
> 狀態：**已成功跑通**

---

## 1. 實驗目標

使用 `PyTorch` 實作的 `LSTM` 模型，在 Sentiment140 **全量資料（160 萬筆）** 上進行情感二分類訓練，完整跑滿 **10 個 epoch**，並在終端顯示每輪訓練進度（`tqdm` 進度條 + 每輪彙總指標）。

---

## 2. 實驗環境

| 項目 | 內容 |
|------|------|
| 作業系統 | Windows 10 |
| Python 環境 | Conda `dl` |
| Python 路徑 | `C:\Users\89275\miniconda3\envs\dl\python.exe` |
| PyTorch | `2.7.1+cu118` |
| CUDA | 可用（`Torch CUDA available: True`） |
| 訓練裝置 | `cuda` |
| 主要依賴 | `torchvision`、`numpy`、`pandas`、`matplotlib`、`scikit-learn`、`tqdm` |

---

## 3. 資料集與預處理

### 3.1 資料來源

- 資料集：**Sentiment140**（Twitter 推文情感資料）
- 原始檔案：`data/training.1600000.processed.noemoticon.csv`
- 預處理檔案：`data/train-processed.csv`（優先使用）

### 3.2 標籤映射

- 原始標籤：`0`（負面）、`4`（正面）
- 映射後：`0` → 負面，`4` → 正面（`1`）

### 3.3 本次載入統計

| 項目 | 數值 |
|------|------|
| 載入樣本數 | **1,600,000**（全量，不抽樣） |
| 負面（label=0）比例 | 0.50 |
| 正面（label=1）比例 | 0.50 |
| 詞彙表大小 | 20,000 |
| 最大序列長度 | 50 |

### 3.4 資料切分

| 集合 | 樣本數 | 比例 |
|------|--------|------|
| 訓練集（Train） | 1,280,000 | 80% |
| 驗證集（Val） | 160,000 | 10% |
| 測試集（Test） | 160,000 | 10% |

- 切分方式：`sklearn.model_selection.train_test_split`（分層抽樣，`stratify=labels`）
- 隨機種子：`seed=42`

### 3.5 文本預處理流程

1. 轉小寫
2. 正則分詞：`[a-z0-9']+`
3. 截斷至 `max_len=50`
4. 未知詞映射為 `<unk>`，填充為 `<pad>`
5. 詞頻低於 `min_freq=2` 的詞不納入詞彙表

---

## 4. 模型結構

### 4.1 架構：`LSTMClassifier`

```
Input (token ids)
    ↓
Embedding(vocab_size=20000, embedding_dim=200, padding_idx=0)
    ↓
LSTM(input_size=200, hidden_size=128, num_layers=1, batch_first=True)
    ↓
Dropout(p=0.2)
    ↓
Linear(128 → 2)
    ↓
Output (logits, 2-class)
```

### 4.2 模型參數說明

| 層 | 參數 |
|----|------|
| Embedding | `vocab_size=20000`, `embedding_dim=200`, `padding_idx=0` |
| LSTM | `hidden_size=128`, `num_layers=1`, `batch_first=True` |
| Dropout | `p=0.2` |
| 全連接層 | `Linear(128, 2)` |

### 4.3 損失函數與優化器

| 項目 | 設定 |
|------|------|
| 損失函數 | `CrossEntropyLoss` |
| 優化器 | `Adam` |
| 學習率 | **0.001** |
| 模型保存策略 | 以驗證集 `val_loss` 最低為準，保存 `best_lstm.pt` |

---

## 5. 超參數總表

| 參數 | 值 | 說明 |
|------|-----|------|
| `epochs` | **10** | 訓練輪次 |
| `batch_size` | 256 | 批次大小 |
| `embedding_dim` | 200 | 詞嵌入維度 |
| `hidden_size` | 128 | LSTM 隱藏層維度 |
| `dropout` | 0.2 | Dropout 比例 |
| `lr` | **0.001** | 學習率 |
| `max_vocab_size` | 20000 | 詞彙表上限 |
| `min_freq` | 2 | 最小詞頻 |
| `max_len` | 50 | 最大序列長度 |
| `max_samples` | -1 | 全量資料（不抽樣） |
| `val_size` | 0.1 | 驗證集比例 |
| `test_size` | 0.1 | 測試集比例 |
| `seed` | 42 | 隨機種子 |

---

## 6. 執行命令

```powershell
conda activate dl
cd pytorch_sentiment_analysis-master\情感分类实战
python src/Pytorch_LSTM实战情感分类.py --epochs 10 --max-samples -1
```

或使用 `conda run`（無需手動 activate）：

```powershell
conda run -n dl python "src/Pytorch_LSTM实战情感分类.py" --epochs 10 --max-samples -1
```

---

## 7. 終端訓練輸出說明

訓練過程中，終端會顯示兩層資訊：

### 7.1 批次進度條（tqdm）

每個 epoch 的 train / val / test 階段都會顯示即時進度條，例如：

```
Epoch 01/10 [train]:  45%|████▌     | 2231/5000 [00:24<00:29, 95.02it/s, acc=0.8329, loss=0.3713]
Epoch 01/10 [val]:   100%|██████████| 625/625 [00:03<00:00, 180.5it/s, acc=0.8234, loss=0.3891]
```

### 7.2 每輪彙總行

每輪結束後印出完整指標：

```
Epoch 01 | train_loss=0.4246, train_acc=0.8032 | val_loss=0.3891, val_acc=0.8234 | test_loss=0.3873, test_acc=0.8249
```

---

## 8. 訓練過程完整記錄（10 輪）

| Epoch | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc |
|:-----:|-----------:|----------:|---------:|--------:|----------:|---------:|
| 1 | 0.4246 | 0.8032 | 0.3891 | 0.8234 | 0.3873 | 0.8249 |
| 2 | 0.3700 | 0.8338 | 0.3777 | 0.8301 | 0.3747 | 0.8321 |
| 3 | 0.3461 | 0.8464 | **0.3758** | **0.8317** | **0.3731** | **0.8332** |
| 4 | 0.3251 | 0.8571 | 0.3805 | 0.8315 | 0.3780 | 0.8329 |
| 5 | 0.3044 | 0.8676 | 0.3880 | 0.8314 | 0.3865 | 0.8319 |
| 6 | 0.2842 | 0.8778 | 0.3968 | 0.8291 | 0.3951 | 0.8309 |
| 7 | 0.2644 | 0.8874 | 0.4215 | 0.8268 | 0.4186 | 0.8275 |
| 8 | 0.2460 | 0.8963 | 0.4346 | 0.8254 | 0.4321 | 0.8270 |
| 9 | 0.2296 | 0.9044 | 0.4572 | 0.8218 | 0.4531 | 0.8239 |
| 10 | 0.2155 | 0.9108 | 0.4751 | 0.8210 | 0.4698 | 0.8228 |

> **粗體** 標示最佳驗證集表現（Epoch 3）

---

## 9. 關鍵結果分析

### 9.1 最佳模型（以 val_loss 為準，Epoch 3）

| 指標 | 數值 |
|------|------|
| val_loss | **0.3758** |
| val_acc | **0.8317** |
| test_loss | **0.3731** |
| test_acc | **0.8332** |

### 9.2 最終輪次（Epoch 10）

| 指標 | 數值 |
|------|------|
| train_acc | 0.9108 |
| val_acc | 0.8210 |
| test_acc | 0.8228 |

### 9.3 訓練曲線趨勢

- **訓練集**：`train_acc` 從 0.8032 持續上升至 0.9108，模型持續學習訓練資料。
- **驗證/測試集**：在第 3 輪達到峰值（`test_acc ≈ 0.8332`），之後緩慢下降。
- **過擬合跡象**：Epoch 3 之後，`train_acc` 持續上升但 `val_acc` / `test_acc` 下降，呈現典型過擬合。

### 9.4 訓練耗時

| 項目 | 時間 |
|------|------|
| 開始時間 | 2026-07-02 12:16:35 |
| 結束時間 | 2026-07-02 12:27:39 |
| 總耗時 | **約 11.07 分鐘**（GPU） |

---

## 10. 輸出檔案清單

| 檔案路徑 | 說明 |
|----------|------|
| `output/models/best_lstm.pt` | 最佳模型權重（Epoch 3，val_loss 最低） |
| `output/logs/training_history.csv` | 每輪 train/val/test loss 與 acc |
| `output/logs/train_full_e10_console.log` | 完整終端訓練日誌（含 tqdm 進度） |
| `output/figures/lstm_loss.png` | Loss 曲線圖 |
| `output/figures/lstm_accuracy.png` | Accuracy 曲線圖 |

---

## 11. 結論

1. 本次實驗已在 Conda `dl` 環境（CUDA GPU）下**成功跑通** Sentiment140 全量 160 萬筆資料，完整訓練 10 輪。
2. 模型在 **Epoch 3** 達到最佳泛化表現：`test_acc = 0.8332`，`val_acc = 0.8317`。
3. 後續輪次（Epoch 4–10）出現過擬合：訓練集準確率持續上升，但驗證/測試集準確率緩慢下降。
4. **建議採用 Epoch 3 的 `best_lstm.pt` 作為最終模型**，而非 Epoch 10 的模型。

---

## 12. 後續優化建議

| 方向 | 具體做法 |
|------|----------|
| 早停（Early Stopping） | 監控 `val_loss`，patience=2~3，避免過擬合 |
| 學習率衰減 | 使用 `ReduceLROnPlateau`，驗證 loss 停滯時降低 lr |
| 正則化加強 | 增大 dropout（0.3~0.5）或加入 weight decay |
| 文本預處理 | 去除 URL、@mention，保留否定詞模式 |
| 模型對照 | 嘗試雙向 LSTM（BiLSTM）或 Transformer baseline |
| 詞向量初始化 | 使用預訓練 GloVe（`data/glove.6B.300d.txt` 已備妥）初始化 Embedding |

---

## 13. 可重現步驟

```powershell
# 1. 啟用環境
conda activate dl

# 2. 進入專案目錄
cd pytorch_sentiment_analysis-master\情感分类实战

# 3. 執行全量 10 輪訓練
python src/Pytorch_LSTM实战情感分类.py --epochs 10 --max-samples -1

# 4. 查看結果
# - 每輪指標：output/logs/training_history.csv
# - 最佳模型：output/models/best_lstm.pt
# - 訓練曲線：output/figures/lstm_loss.png, lstm_accuracy.png
```
