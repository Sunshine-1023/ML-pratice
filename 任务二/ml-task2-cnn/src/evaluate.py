"""CIFAR-10 測試腳本。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import build_model_from_config, get_model_config, normalize_model_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


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


def get_test_loader(batch_size: int = 128, num_workers: int = 2) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_root, should_download = resolve_cifar10_root()
    print(f"資料路徑: {cifar_root} (download={should_download})")
    test_set = datasets.CIFAR10(root=str(cifar_root), train=False, download=should_download, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[int]]:
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for images, labels in tqdm(loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for label, pred in zip(labels, predicted):
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1

    accuracy = correct / total
    return accuracy, [c / t if t > 0 else 0.0 for c, t in zip(class_correct, class_total)]


def main() -> None:
    parser = argparse.ArgumentParser(description="評估 CIFAR-10 CNN 模型")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型權重路徑，如 outputs/models/base_cnn_best.pth")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        # 先嘗試使用目前工作目錄的相對路徑，再相容舊用法（相對專案根目錄）
        if ckpt_path.exists():
            ckpt_path = ckpt_path.resolve()
        else:
            ckpt_path = (PROJECT_ROOT / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint.get("model_name", "base_cnn")
    model_config = checkpoint.get("model_config")
    if model_config is None:
        # 相容舊 checkpoint：由預設配置重建
        model_config = get_model_config(model_name)
        model_name = normalize_model_name(model_name)
    model = build_model_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = get_test_loader(batch_size=args.batch_size)
    accuracy, per_class_acc = evaluate_model(model, test_loader, device)

    print(f"模型: {model_name}")
    print(f"模型配置: {model_config}")
    print(f"檢查點: {ckpt_path}")
    print(f"整體測試準確率: {accuracy * 100:.2f}%")
    print("\n各類別準確率:")
    for cls_name, acc in zip(CIFAR10_CLASSES, per_class_acc):
        print(f"  {cls_name:12s}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
