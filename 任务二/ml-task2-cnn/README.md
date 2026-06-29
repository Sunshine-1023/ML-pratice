# ml-task2-cnn

基於 PyTorch 的 CIFAR-10 卷積神經網路（CNN）訓練與評估專案，包含三種模型架構對比實驗。

## 專案結構

```
ml-task2-cnn/
├── data/                  # CIFAR-10 資料集自動下載到此
├── outputs/
│   ├── figures/           # loss / accuracy 曲線圖
│   ├── models/            # 訓練好的模型權重
│   └── logs/              # CSV 訓練日誌
├── src/
│   ├── models.py          # 三種 CNN 模型
│   ├── train.py           # 訓練程式
│   ├── evaluate.py        # 測試程式
│   └── plot_results.py    # 繪圖程式
├── requirements.txt
└── README.md
```

## 模型說明

| 模型名稱 | 說明 |
|---------|------|
| `base_cnn` | 模型1：基礎 CNN（標準深度與通道數） |
| `wide_cnn` | 模型2：加寬 CNN（保持深度，增加卷積通道數） |
| `deep_cnn` | 模型3：加深 CNN（增加卷積層數） |

## 環境安裝

```bash
cd ml-task2-cnn
pip install -r requirements.txt
```

## 使用方法

### 0. 一鍵跑完整實驗（推薦）

```bash
cd ml-task2-cnn
python run_all_experiments.py --epochs 20 --device auto
```

此命令會自動完成三模型訓練、評估、出圖，並生成 `outputs/logs/experiment_summary.csv`。

### 1. 訓練模型

```bash
cd src

# 訓練模型1：基礎 CNN
python train.py --model base_cnn --epochs 20

# 訓練模型2：加寬 CNN
python train.py --model wide_cnn --epochs 20

# 訓練模型3：加深 CNN
python train.py --model deep_cnn --epochs 20

# 自訂參數（示例：5x5 卷積核、加寬通道、每個 stage 2 層、啟用 BN、Dropout=0.3）
python train.py --model base_cnn \
  --kernel-size 5 \
  --channels 64,128,256 \
  --stage-depths 2,2,2 \
  --use-batchnorm \
  --dropout 0.3
```

訓練完成後會自動：
- 下載 CIFAR-10 至 `data/`
- 保存最佳模型至 `outputs/models/{model}_best.pth`
- 保存訓練日誌至 `outputs/logs/{model}_train_log.csv`

### 2. 評估模型

```bash
cd src
python evaluate.py --checkpoint ../outputs/models/base_cnn_best.pth
```

### 3. 繪製訓練曲線

```bash
cd src

# 繪製單一模型曲線
python plot_results.py --log ../outputs/logs/base_cnn_train_log.csv

# 比較所有模型的測試準確率
python plot_results.py --compare

# 一次輸出「每個模型的 loss/accuracy 曲線」+「三模型對比圖」
python plot_results.py --all
```

圖片輸出至 `outputs/figures/`。

## 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `base_cnn` | 模型名稱（`base_cnn` / `wide_cnn` / `deep_cnn`） |
| `--kernel-size` | 模型預設值 | 卷積核大小（例如 3、5） |
| `--channels` | 模型預設值 | 各 stage 通道（例如 `32,64,128`） |
| `--stage-depths` | 模型預設值 | 各 stage 卷積層數（例如 `1,1,1`） |
| `--use-batchnorm` / `--no-batchnorm` | 模型預設值 | 強制開/關 BatchNorm |
| `--use-dropout` / `--no-dropout` | 模型預設值 | 強制開/關 Dropout |
| `--dropout` | 模型預設值 | Dropout 機率（例如 `0.3`） |
| `--epochs` | `20` | 訓練輪數 |
| `--batch-size` | `128` | 批次大小 |
| `--lr` | `0.001` | 學習率 |
| `--device` | `auto` | 裝置（auto / cuda / mps / cpu） |

## 輸出說明

- **models/**：包含 `model_state_dict`、模型名稱、最佳 epoch 與測試準確率（`test_acc`）
- **logs/**：每個 epoch 的 `train_loss`、`train_acc`、`test_loss`、`test_acc`
- **figures/**：每個模型各自的 Loss/Accuracy 曲線圖，以及三模型 Test Accuracy 對比圖
