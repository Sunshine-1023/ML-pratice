"""CIFAR-10 訓練腳本。"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import build_model, get_model_config, normalize_model_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
TRAINABLE_MODEL_NAMES = ["base_cnn", "wide_cnn", "deep_cnn"]


def parse_int_list(raw: str, arg_name: str) -> list[int]:
    """解析逗號分隔整數列表，例如 '32,64,128'。"""
    try:
        values = [int(x.strip()) for x in raw.split(",")]
    except ValueError as exc:
        raise ValueError(f"{arg_name} 需為逗號分隔整數，例如 32,64,128") from exc
    if not values:
        raise ValueError(f"{arg_name} 不能為空")
    return values


def resolve_cifar10_root() -> tuple[Path, bool]:
    """優先使用本地已解壓資料，找不到才啟用下載。"""
    candidates = [
        DATA_DIR,
        DATA_DIR / "cifar-10-python(加密)" / "cifar-10-python",
    ]
    for root in candidates:
        if (root / "cifar-10-batches-py").exists():
            return root, False
    return DATA_DIR, True


def get_dataloaders(batch_size: int = 128, num_workers: int = 2) -> tuple[DataLoader, DataLoader]:
    """載入 CIFAR-10 訓練集與測試集。"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_root, should_download = resolve_cifar10_root()
    print(f"資料路徑: {cifar_root} (download={should_download})")

    train_set = datasets.CIFAR10(root=str(cifar_root), train=True, download=should_download, transform=transform_train)
    test_set = datasets.CIFAR10(root=str(cifar_root), train=False, download=should_download, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def save_log(log_path: Path, rows: list[dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "time_sec"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="訓練 CIFAR-10 CNN 模型")
    parser.add_argument("--model", type=str, default="base_cnn", choices=TRAINABLE_MODEL_NAMES)
    parser.add_argument("--kernel-size", type=int, default=None, help="覆蓋卷積核大小，例如 3 或 5")
    parser.add_argument("--channels", type=str, default=None, help="覆蓋各 stage 通道，例如 32,64,128")
    parser.add_argument("--stage-depths", type=str, default=None, help="覆蓋各 stage 卷積層數，例如 1,1,1")
    parser.add_argument("--dropout", type=float, default=None, help="覆蓋 Dropout 機率，例如 0.3")
    parser.add_argument("--use-batchnorm", action="store_true", help="強制啟用 BatchNorm")
    parser.add_argument("--no-batchnorm", action="store_true", help="強制關閉 BatchNorm")
    parser.add_argument("--use-dropout", action="store_true", help="強制啟用 Dropout")
    parser.add_argument("--no-dropout", action="store_true", help="強制關閉 Dropout")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.use_batchnorm and args.no_batchnorm:
        raise ValueError("--use-batchnorm 與 --no-batchnorm 不能同時指定")
    if args.use_dropout and args.no_dropout:
        raise ValueError("--use-dropout 與 --no-dropout 不能同時指定")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model_overrides: dict = {}
    if args.kernel_size is not None:
        model_overrides["kernel_size"] = args.kernel_size
    if args.channels:
        model_overrides["channels"] = parse_int_list(args.channels, "--channels")
    if args.stage_depths:
        model_overrides["stage_depths"] = parse_int_list(args.stage_depths, "--stage-depths")
    if args.dropout is not None:
        model_overrides["dropout_p"] = args.dropout
    if args.use_batchnorm:
        model_overrides["use_batchnorm"] = True
    elif args.no_batchnorm:
        model_overrides["use_batchnorm"] = False
    if args.use_dropout:
        model_overrides["use_dropout"] = True
    elif args.no_dropout:
        model_overrides["use_dropout"] = False

    print(f"使用裝置: {device}")
    print(f"模型: {args.model}")
    if model_overrides:
        print(f"參數覆蓋: {model_overrides}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = build_model(args.model, config_overrides=model_overrides).to(device)
    model_config = get_model_config(args.model, config_overrides=model_overrides)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = normalize_model_name(args.model)
    if model_overrides:
        model_tag = f"{model_tag}_custom"
    log_path = LOG_DIR / f"{model_tag}_train_log.csv"
    best_test_acc = 0.0
    log_rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "time_sec": round(elapsed, 2),
        })
        save_log(log_path, log_rows)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"time={elapsed:.1f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            ckpt_path = MODEL_DIR / f"{model_tag}_best.pth"
            torch.save({
                "model_name": normalize_model_name(args.model),
                "model_config": model_config,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc,
                # 兼容舊欄位名稱
                "val_acc": test_acc,
            }, ckpt_path)
            print(f"  -> 保存最佳模型: {ckpt_path} (test_acc={test_acc:.4f})")

    print(f"訓練完成，最佳測試準確率: {best_test_acc:.4f}")
    print(f"日誌已保存至: {log_path}")


if __name__ == "__main__":
    main()
