# MNIST 手寫數字辨識實驗記錄

## 1. 實驗概述

本次實驗使用 PyTorch 在 MNIST 手寫數字資料集上訓練兩個卷積神經網路模型，分別為基礎模型 `SimpleCNN` 與加強模型 `BetterCNN`。實驗目標是比較兩種 CNN 結構在 MNIST 十分類任務上的訓練收斂情況、測試集準確率、loss 變化，並驗證訓練後模型能否對手寫數字圖片完成推論。

正式實驗已完整跑通，包含資料讀取、模型訓練、指標記錄、圖表生成、模型保存與圖片推論驗證。

## 2. 實驗環境

| 項目 | 值 |
| --- | --- |
| 作業系統 | macOS-26.3.1-arm64-arm-64bit |
| Conda 環境 | `ai` |
| Python | 3.10.20 |
| Python 路徑 | `/opt/miniconda3/envs/ai/bin/python` |
| PyTorch | 2.11.0 |
| torchvision | 0.26.0 |
| NumPy | 2.2.6 |
| Matplotlib | 3.10.9 |
| Pillow | 12.2.0 |
| CUDA | 不可用 |
| Apple MPS | 可用 |
| 正式訓練裝置 | `mps` |

注意：系統預設 `python` 指向 pyenv 的 `/Users/sunshine/.pyenv/versions/3.11.6/bin/python`，該環境缺少 `_lzma`，會導致 `torchvision` 匯入失敗。因此正式實驗使用 README 建議的 conda `ai` 環境執行。

## 3. 專案與程式結構

| 檔案 | 用途 |
| --- | --- |
| `src/models.py` | 定義 `SimpleCNN`、`BetterCNN` 與 `build_model()` |
| `src/train.py` | 訓練入口，負責資料載入、訓練、測試、保存模型與 CSV |
| `src/plot_results.py` | 讀取訓練 CSV，生成 loss / accuracy 對比圖 |
| `src/predict_image.py` | 對外部圖片做預處理並載入模型推論 |
| `run_experiment.py` | 一鍵執行完整實驗：訓練兩個模型並畫圖，可選擇推論圖片 |
| `EXPERIMENT_RECORD.md` | 本次正式實驗記錄 |

## 4. 資料集與資料處理

| 項目 | 值 |
| --- | --- |
| 資料集 | MNIST |
| 來源 | `torchvision.datasets.MNIST` |
| 訓練集數量 | 60000 |
| 測試集數量 | 10000 |
| 類別數 | 10，數字 `0` 到 `9` |
| 圖片大小 | `1 x 28 x 28` |
| 下載位置 | `data/` |
| 訓練集 shuffle | 是 |
| 測試集 shuffle | 否 |

資料前處理如下：

```python
transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
```

其中 `0.1307` 和 `0.3081` 是 MNIST 常用的灰階均值與標準差。訓練與圖片推論都使用相同 normalization，確保輸入分佈一致。

## 5. 模型方法

### 5.1 SimpleCNN

`SimpleCNN` 是基礎 CNN 對照模型，結構如下：

| 模組 | 結構 |
| --- | --- |
| 特徵提取 | `Conv2d(1, 16, 3, padding=1)` + ReLU + MaxPool |
| 特徵提取 | `Conv2d(16, 32, 3, padding=1)` + ReLU + MaxPool |
| 分類器 | Flatten |
| 分類器 | `Linear(32 * 7 * 7, 128)` + ReLU |
| 輸出層 | `Linear(128, 10)` |
| 總參數量 | 206922 |
| 可訓練參數量 | 206922 |

### 5.2 BetterCNN

`BetterCNN` 是加強模型，加入 BatchNorm 與 Dropout，並提高卷積通道數：

| 模組 | 結構 |
| --- | --- |
| 特徵提取 | `Conv2d(1, 32, 3, padding=1)` + BatchNorm + ReLU |
| 特徵提取 | `Conv2d(32, 64, 3, padding=1)` + BatchNorm + ReLU + MaxPool |
| 正則化 | `Dropout2d(p=0.25)` |
| 特徵提取 | `Conv2d(64, 128, 3, padding=1)` + ReLU + MaxPool |
| 分類器 | Flatten |
| 分類器 | `Linear(128 * 7 * 7, 256)` + ReLU |
| 正則化 | `Dropout(p=0.5)` |
| 輸出層 | `Linear(256, 10)` |
| 總參數量 | 1701322 |
| 可訓練參數量 | 1701322 |

## 6. 訓練設定

| 項目 | 值 |
| --- | --- |
| Epochs | 10 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Random seed | 42 |
| DataLoader workers | 2 |
| 訓練裝置 | Apple MPS |
| 評價資料 | MNIST test set |

本專案每個 epoch 都在完整訓練集上訓練一次，然後在完整測試集上計算 `test_loss` 與 `test_acc`。這是課程實驗常見流程；嚴格機器學習實驗中通常還會額外切出 validation set，但本專案未設 validation split。

## 7. 正式執行命令

正式實驗在專案根目錄執行：

```bash
cd "/Users/sunshine/Desktop/小学期实践/任务二/mnist_task"
MPLCONFIGDIR="$(pwd)/outputs/matplotlib" conda run -n ai python run_experiment.py --epochs 10
```

`run_experiment.py` 實際依序執行：

```bash
/opt/miniconda3/envs/ai/bin/python src/train.py --model simple_cnn --epochs 10 --batch-size 128 --lr 0.001 --seed 42
/opt/miniconda3/envs/ai/bin/python src/train.py --model better_cnn --epochs 10 --batch-size 128 --lr 0.001 --seed 42
/opt/miniconda3/envs/ai/bin/python src/plot_results.py
```

完整終端日誌已保存：

```text
outputs/logs/experiment_run.log
```

## 8. 評價指標

本次實驗記錄四個主要指標：

| 指標 | 說明 |
| --- | --- |
| `train_loss` | 訓練集平均交叉熵損失 |
| `train_acc` | 訓練集分類準確率 |
| `test_loss` | 測試集平均交叉熵損失 |
| `test_acc` | 測試集分類準確率 |

準確率計算方式為：

```text
accuracy = 預測正確樣本數 / 總樣本數
```

## 9. SimpleCNN 訓練結果

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.221795 | 93.4417% | 0.068683 | 97.6800% |
| 2 | 0.059800 | 98.1800% | 0.039671 | 98.7500% |
| 3 | 0.042273 | 98.6917% | 0.033649 | 98.8500% |
| 4 | 0.032595 | 98.9617% | 0.035304 | 98.8100% |
| 5 | 0.025149 | 99.2083% | 0.032912 | 98.8300% |
| 6 | 0.020090 | 99.3683% | 0.039801 | 98.7500% |
| 7 | 0.016363 | 99.4550% | 0.039356 | 98.8100% |
| 8 | 0.011383 | 99.6283% | 0.035522 | 98.7700% |
| 9 | 0.010536 | 99.6650% | 0.031775 | 99.0000% |
| 10 | 0.010397 | 99.6583% | 0.036420 | 98.9000% |

SimpleCNN 最終結果：

| 指標 | 值 |
| --- | ---: |
| 最終 train_loss | 0.010397 |
| 最終 train_acc | 99.6583% |
| 最終 test_loss | 0.036420 |
| 最終 test_acc | 98.9000% |
| 最高 test_acc | 99.0000%，第 9 輪 |
| 最低 test_loss | 0.031775，第 9 輪 |

## 10. BetterCNN 訓練結果

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.201943 | 93.8450% | 0.042918 | 98.5000% |
| 2 | 0.064241 | 98.1050% | 0.032999 | 98.8800% |
| 3 | 0.047709 | 98.5600% | 0.024383 | 99.1700% |
| 4 | 0.040015 | 98.7900% | 0.027404 | 99.1000% |
| 5 | 0.033068 | 98.9617% | 0.022508 | 99.2700% |
| 6 | 0.030268 | 98.9883% | 0.022473 | 99.2100% |
| 7 | 0.025657 | 99.2150% | 0.025681 | 99.2100% |
| 8 | 0.024075 | 99.2017% | 0.020048 | 99.3200% |
| 9 | 0.020018 | 99.3683% | 0.021083 | 99.4000% |
| 10 | 0.019997 | 99.3550% | 0.021915 | 99.3400% |

BetterCNN 最終結果：

| 指標 | 值 |
| --- | ---: |
| 最終 train_loss | 0.019997 |
| 最終 train_acc | 99.3550% |
| 最終 test_loss | 0.021915 |
| 最終 test_acc | 99.3400% |
| 最高 test_acc | 99.4000%，第 9 輪 |
| 最低 test_loss | 0.020048，第 8 輪 |

## 11. 模型對比

| 模型 | 參數量 | 最終 Test Acc | 最高 Test Acc | 最終 Test Loss |
| --- | ---: | ---: | ---: | ---: |
| SimpleCNN | 206922 | 98.9000% | 99.0000% | 0.036420 |
| BetterCNN | 1701322 | 99.3400% | 99.4000% | 0.021915 |

對比結果：

- `BetterCNN` 的最終測試準確率比 `SimpleCNN` 高 `0.44` 個百分點。
- `BetterCNN` 的最高測試準確率比 `SimpleCNN` 高 `0.40` 個百分點。
- `BetterCNN` 的最終測試 loss 更低，說明其在測試集上的分類信心與擬合效果更好。
- `SimpleCNN` 的參數量更小，訓練後也達到 98.90% 測試準確率，作為基礎模型已經足夠有效。
- `BetterCNN` 使用 BatchNorm 與 Dropout，在更大模型容量下仍保持較好的泛化能力。

## 12. 輸出檔案驗證

| 輸出檔案 | 驗證結果 |
| --- | --- |
| `outputs/logs/simple_cnn_history.csv` | 存在，10 筆 epoch 記錄 |
| `outputs/logs/better_cnn_history.csv` | 存在，10 筆 epoch 記錄 |
| `outputs/logs/experiment_run.log` | 存在，保存正式訓練終端輸出 |
| `outputs/models/simple_cnn.pth` | 存在，831117 bytes，checkpoint 可載入 |
| `outputs/models/better_cnn.pth` | 存在，6812775 bytes，checkpoint 可載入 |
| `outputs/figures/train_loss_compare.png` | 存在，1200 x 750 |
| `outputs/figures/test_loss_compare.png` | 存在，1200 x 750 |
| `outputs/figures/train_accuracy_compare.png` | 存在，1200 x 750 |
| `outputs/figures/test_accuracy_compare.png` | 存在，1200 x 750 |
| `outputs/figures/final_test_accuracy_compare.png` | 存在，900 x 750 |
| `outputs/figures/mnist_sample_7_preprocess.png` | 存在，2250 x 450 |

## 13. 圖片推論驗證

為了驗證 `predict_image.py` 的完整流程，從 MNIST 測試集取出一張標籤為 `7` 的樣本，保存為：

```text
images/mnist_sample_7.png
```

使用正式訓練後的 `better_cnn.pth` 推論：

```bash
MPLCONFIGDIR="$(pwd)/outputs/matplotlib" conda run -n ai python src/predict_image.py --image images/mnist_sample_7.png --model outputs/models/better_cnn.pth
```

推論結果：

| 項目 | 值 |
| --- | --- |
| 測試圖片 | `images/mnist_sample_7.png` |
| 真實標籤 | 7 |
| 使用模型 | `outputs/models/better_cnn.pth` |
| 預測結果 | 7 |
| 置信度 | 1.0000 |
| 預處理流程圖 | `outputs/figures/mnist_sample_7_preprocess.png` |
| 推論日誌 | `outputs/logs/predict_sample.log` |

推論過程出現 Matplotlib 中文字型 warning，原因是目前預設字型 DejaVu Sans 缺少部分中文字形。這不影響模型推論結果，也不影響圖片檔案生成，只可能讓預處理流程圖中的中文標題顯示不完整。

## 14. 結論

本次 MNIST 實驗已完整跑通。兩個 CNN 模型都能正常訓練、保存模型、記錄 CSV 指標並生成對比圖。`BetterCNN` 在最終測試準確率、最高測試準確率與最終測試 loss 上均優於 `SimpleCNN`，說明加入 BatchNorm、Dropout 與更大通道數後，模型在 MNIST 任務上取得了更好的泛化表現。

最終推薦使用 `BetterCNN` 作為圖片推論模型，其正式實驗最終測試準確率為 `99.34%`，最高測試準確率為 `99.40%`。

