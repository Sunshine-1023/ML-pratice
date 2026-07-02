# Sentiment140 全量實驗記錄（LSTM，lr=0.0001，batch=512，ES=2）

## 1. 實驗目標
- 重新啟動並完成一次全量資料訓練。
- 使用獨立輸出目錄，避免覆蓋既有實驗結果。
- 記錄本次環境、參數、訓練指標與產出檔案。

## 2. 執行資訊
- `run_name`: `lstm_full_bs512_lr1e4_es2_20260702_1824`
- 啟動命令：
  - `python -u "src/run_lstm_experiment_custom_output.py" --run-name "lstm_full_bs512_lr1e4_es2_20260702_1824" --epochs 10 --max-samples -1`
- 開始時間（UTC）：`2026-07-02T10:23:24.438Z`
- 結束時間（UTC）：`2026-07-02T10:40:05.761Z`
- 總耗時：`1001323 ms`（約 `16 分 41 秒`）
- 退出狀態：`exit_code = 0`

## 3. 環境與裝置
- Python 環境：`conda dl`
- PyTorch CUDA 可用：`True`
- 實際裝置：`cuda`
- 資料量：`1600000`
- 資料切分：`Train / Val / Test = 1280000 / 160000 / 160000`
- GloVe 覆蓋：`matched=16586, coverage=82.94%`

## 4. 主要訓練參數
- `epochs=10`
- `lr=0.0001`
- `batch_size=512`
- `embedding_dim=300`
- `hidden_size=128`
- `num_layers=2`
- `dropout=0.2`
- `max_samples=-1`（全量）
- 早停：
  - `early_stop_patience=2`
  - `early_stop_min_delta=0.0`
  - 本次未提前停止（實際跑滿 10 輪）

## 5. 每輪指標（training_history.csv）
| epoch | train_loss | val_loss | test_loss | train_acc | val_acc | test_acc |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4579 | 0.4206 | 0.4201 | 0.7827 | 0.8069 | 0.8058 |
| 2 | 0.4034 | 0.3986 | 0.3977 | 0.8157 | 0.8190 | 0.8199 |
| 3 | 0.3871 | 0.3907 | 0.3889 | 0.8243 | 0.8225 | 0.8245 |
| 4 | 0.3767 | 0.3868 | 0.3851 | 0.8300 | 0.8255 | 0.8272 |
| 5 | 0.3690 | 0.3821 | 0.3798 | 0.8340 | 0.8272 | 0.8297 |
| 6 | 0.3626 | 0.3800 | 0.3778 | 0.8377 | 0.8290 | 0.8309 |
| 7 | 0.3567 | 0.3789 | 0.3769 | 0.8408 | 0.8292 | 0.8300 |
| 8 | 0.3513 | 0.3781 | 0.3759 | 0.8436 | 0.8306 | 0.8330 |
| 9 | 0.3462 | 0.3785 | 0.3757 | 0.8462 | 0.8301 | 0.8324 |
| 10 | 0.3412 | 0.3772 | 0.3747 | 0.8490 | 0.8309 | 0.8322 |

## 6. 關鍵結果摘要
- 最佳驗證損失（`val_loss` 最低）：`0.3772`（Epoch 10）
- 最佳測試準確率（`test_acc` 最高）：`0.8330`（Epoch 8）
- 最終（Epoch 10）：
  - `val_acc=0.8309`
  - `test_acc=0.8322`

## 7. 輸出檔案（獨立，不覆蓋舊結果）
- 實驗根目錄：
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/`
- 圖表：
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/figures/lstm_loss.png`
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/figures/lstm_accuracy.png`
- 指標：
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/logs/training_history.csv`
- 完整終端日誌：
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/logs/train_console_full.log`
- 最佳模型：
  - `output/experiments/lstm_full_bs512_lr1e4_es2_20260702_1824/models/best_lstm.pt`
