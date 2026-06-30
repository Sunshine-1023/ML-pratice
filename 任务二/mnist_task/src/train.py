import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from models import build_model
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN models on MNIST.")
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "better_cnn"],
        required=True,
        help="要訓練的模型名稱。",
    )
    parser.add_argument("--epochs", type=int, default=10, help="訓練輪數。")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小。")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率。")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子。")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def save_history(model_name: str, history: list[dict[str, float]]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{model_name}_history.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(history)
    return log_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_loader, test_loader = get_dataloaders(args.batch_size)
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []

    print(f"使用裝置：{device}")
    print(f"開始訓練：{args.model}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        with torch.no_grad():
            test_loss, test_acc = run_one_epoch(model, test_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

    model_path = MODEL_DIR / f"{args.model}.pth"
    torch.save(
        {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "test_acc": history[-1]["test_acc"],
        },
        model_path,
    )
    log_path = save_history(args.model, history)

    print(f"模型已儲存：{model_path}")
    print(f"訓練紀錄已儲存：{log_path}")


if __name__ == "__main__":
    main()
