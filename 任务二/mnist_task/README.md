# MNIST 手寫數字辨識實驗

本專案完成兩個 CNN 模型在 MNIST 資料集上的訓練、指標紀錄、結果畫圖，以及自己拍攝手寫數字照片後進行預處理與模型預測。

## 專案結構

```text
mnist_task/
├── data/
├── outputs/
│   ├── logs/
│   ├── models/
│   └── figures/
├── images/
├── src/
│   ├── models.py
│   ├── train.py
│   ├── predict_image.py
│   └── plot_results.py
└── README.md
```

## 環境

先啟動 conda 環境：

```bash
conda activate ai
```

若缺少套件，可安裝：

```bash
pip install torch torchvision matplotlib numpy pillow
```

## 訓練模型

訓練基礎模型：

```bash
python src/train.py --model simple_cnn --epochs 10
```

訓練較強模型：

```bash
python src/train.py --model better_cnn --epochs 10
```

訓練完成後會產生：

- `outputs/models/simple_cnn.pth`
- `outputs/models/better_cnn.pth`
- `outputs/logs/simple_cnn_history.csv`
- `outputs/logs/better_cnn_history.csv`

CSV 會記錄每一輪的 `train_loss`、`train_acc`、`test_loss`、`test_acc`。

## 畫結果圖

兩個模型都訓練完後執行：

```bash
python src/plot_results.py
```

輸出圖會放在 `outputs/figures/`：

- `train_loss_compare.png`
- `test_loss_compare.png`
- `train_accuracy_compare.png`
- `test_accuracy_compare.png`
- `final_test_accuracy_compare.png`

## 預測自己拍攝的手寫數字

先把手機拍的圖片放進 `images/`，例如：

```text
images/digit_7.jpg
```

再執行：

```bash
python src/predict_image.py --image images/digit_7.jpg --model outputs/models/better_cnn.pth
```

輸出範例：

```text
預測結果：7
置信度：0.9800
預處理流程圖已儲存：outputs/figures/digit_7_preprocess.png
```

預處理流程包含：

1. 讀取圖片
2. 轉灰階
3. 裁切數字區域
4. 補成正方形
5. 縮放到 `28 × 28`
6. 白底黑字反色成黑底白字
7. 正規化成 MNIST 使用的格式
8. 轉成模型輸入 `1 × 1 × 28 × 28`

## 報告建議

報告可依照以下順序撰寫：

1. 實驗目的
2. 實驗環境
3. MNIST 資料集介紹
4. `SimpleCNN` 與 `BetterCNN` 結構設計
5. 訓練參數設定
6. loss / accuracy 曲線
7. 兩種模型結果對比
8. 自己拍照手寫數字的預處理與預測結果
9. 實驗總結

建議放入的圖片：

- MNIST 資料集樣例圖
- 兩個模型的 loss / accuracy 曲線
- 最終 test accuracy 對比圖
- 自己手寫數字原圖
- 預處理流程圖
- 預測結果截圖
