# 附錄：重點程式碼

本附錄整理兩個實驗的核心程式碼，便於報告中說明「模型如何定義、資料如何處理、訓練如何執行、結果如何記錄」。

---

## 附錄 A：CIFAR-10 三模型對比實驗（`ml-task2-cnn`）

### A.1 三種模型配置

```python
PRESET_MODEL_CONFIGS = {
    "base_cnn": {
        "channels": [32, 64, 128],
        "stage_depths": [1, 1, 1],
        "kernel_size": 3,
        "use_batchnorm": False,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
    "wide_cnn": {
        "channels": [64, 128, 256],
        "stage_depths": [1, 1, 1],
        "kernel_size": 3,
        "use_batchnorm": False,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
    "deep_cnn": {
        "channels": [32, 64, 128],
        "stage_depths": [2, 2, 2],
        "kernel_size": 3,
        "use_batchnorm": True,
        "use_dropout": True,
        "dropout_p": 0.5,
    },
}
```

**說明：**
- `base_cnn`：基礎 CNN
- `wide_cnn`：加寬 CNN（增大通道數）
- `deep_cnn`：加深 CNN（增加卷積層數，並加入 BatchNorm）

---

### A.2 可配置 CNN 模型結構

```python
class ConfigurableCNN(nn.Module):
    def __init__(self, num_classes=10, channels=(32, 64, 128),
                 stage_depths=(1, 1, 1), kernel_size=3,
                 use_batchnorm=False, use_dropout=True, dropout_p=0.5):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        in_channels = 3

        for out_channels, depth in zip(channels, stage_depths):
            for _ in range(depth):
                layers.append(nn.Conv2d(in_channels, out_channels,
                                        kernel_size=kernel_size, padding=padding))
                if use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        classifier = [
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            classifier.append(nn.Dropout(dropout_p))
        classifier.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return self.classifier(x)
```

**說明：**
- 每個 stage 可配置卷積層數 `stage_depths`
- 可選擇是否加入 `BatchNorm` 與 `Dropout`
- 最後輸出 10 類分類結果

---

### A.3 資料載入與預處理

```python
def get_dataloaders(batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10(root=cifar_root, train=True,
                                 download=False, transform=transform_train)
    test_set = datasets.CIFAR10(root=cifar_root, train=False,
                                download=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
```

**說明：**
- 訓練集使用隨機裁剪與水平翻轉做資料增強
- 測試集只做標準化，不做增強

---

### A.4 單輪訓練與測試

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total
```

---

### A.5 訓練主流程與日誌記錄

```python
model = build_model(args.model).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(1, 21):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                            criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader,
                                   criterion, device)
    scheduler.step()

    log_rows.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    })

    if test_acc > best_test_acc:
        torch.save({
            "model_name": args.model,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "test_acc": test_acc,
        }, f"outputs/models/{args.model}_best.pth")
```

**說明：**
- 每輪記錄 `train_loss / train_acc / test_loss / test_acc`
- 保存測試準確率最高的模型權重

---

### A.6 一鍵執行實驗

```bash
cd ml-task2-cnn
python run_all_experiments.py --epochs 20 --device auto
```

---

## 附錄 B：MNIST 手寫數字實驗（`mnist_task`）

### B.1 兩種模型定義

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**說明：**
- `SimpleCNN`：基礎對照模型
- `BetterCNN`：加入 BatchNorm 與 Dropout，提升泛化能力

---

### B.2 MNIST 資料載入

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True,
                               download=True, transform=transform)
test_dataset = datasets.MNIST(root=DATA_DIR, train=False,
                              download=True, transform=transform)
```

---

### B.3 訓練與記錄

```python
for epoch in range(1, args.epochs + 1):
    train_loss, train_acc = run_one_epoch(
        model, train_loader, criterion, device, optimizer)
    with torch.no_grad():
        test_loss, test_acc = run_one_epoch(
            model, test_loader, criterion, device)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    })

torch.save({
    "model_name": args.model,
    "model_state_dict": model.state_dict(),
    "test_acc": history[-1]["test_acc"],
}, f"outputs/models/{args.model}.pth")
```

---

### B.4 自拍照預處理與預測（重點）

```python
def preprocess_image(image_path):
    original = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(original)
    gray = ImageOps.autocontrast(gray)

    # 1. 找到手寫數字區域
    bbox = find_digit_bbox(gray)
    cropped = gray.crop(bbox)

    # 2. 補成正方形
    square = make_square(cropped, fill=255)

    # 3. 縮放到 28x28
    resized = square.resize((28, 28), Image.Resampling.LANCZOS)

    # 4. 轉成 MNIST 風格（黑底白字）
    mnist_like = ImageOps.invert(resized)

    # 5. 正規化並轉成模型輸入
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])(mnist_like).unsqueeze(0)

    return tensor
```

```python
with torch.no_grad():
    outputs = model(image_tensor.to(device))
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = probabilities.max(dim=1)

print(f"預測結果：{prediction.item()}")
print(f"置信度：{confidence.item():.4f}")
```

**說明：**
預處理流程為：原圖 → 灰階 → 裁切 → 正方形 → 28×28 → 反色 → 正規化 → 模型輸入。

---

### B.5 訓練與預測命令

```bash
# 訓練
python src/train.py --model simple_cnn --epochs 10
python src/train.py --model better_cnn --epochs 10

# 畫圖
python src/plot_results.py

# 預測自己拍攝的手寫數字
python src/predict_image.py \
  --image images/digit_7.jpg \
  --model outputs/models/better_cnn.pth
```

---

## 附錄 C：兩個實驗的共同設計思路

| 模組 | 作用 |
|------|------|
| `models.py` | 定義網路結構 |
| `train.py` | 訓練、評估、保存日誌與模型 |
| `plot_results.py` | 繪製 loss / accuracy 曲線 |
| `outputs/logs/` | 保存每輪訓練指標 |
| `outputs/models/` | 保存最佳模型權重 |
| `outputs/figures/` | 保存實驗圖表 |

**實驗記錄欄位統一為：**
- `train_loss`
- `train_acc`
- `test_loss`
- `test_acc`

這些欄位可直接用於報告中的曲線圖與結果分析。

---

## 附錄 D：報告可引用的執行命令

### CIFAR-10 實驗

```bash
conda activate ai
cd ml-task2-cnn
python run_all_experiments.py --epochs 20 --device auto
cd src
python plot_results.py --all
```

### MNIST 實驗

```bash
conda activate ai
cd mnist_task
python src/train.py --model simple_cnn --epochs 10
python src/train.py --model better_cnn --epochs 10
python src/plot_results.py
```

---

*本附錄對應專案路徑：`任务二/ml-task2-cnn` 與 `任务二/mnist_task`*
