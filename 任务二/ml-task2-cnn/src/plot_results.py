"""繪製訓練 loss / accuracy 曲線。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def _pick_metric_col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    """優先使用新欄位名，必要時相容舊欄位名。"""
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"缺少欄位: {preferred} / {fallback}")


def plot_single_log(log_path: Path, output_path: Path | None = None) -> Path:
    """繪製單一訓練日誌的曲線圖。"""
    df = pd.read_csv(log_path)
    model_name = log_path.stem.replace("_train_log", "")
    test_loss_col = _pick_metric_col(df, preferred="test_loss", fallback="val_loss")
    test_acc_col = _pick_metric_col(df, preferred="test_acc", fallback="val_acc")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(df["epoch"], df[test_loss_col], label="Test Loss", marker="s", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[1].plot(df["epoch"], df[test_acc_col], label="Test Acc", marker="s", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIG_DIR / f"{model_name}_curves.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_compare_logs(log_paths: list[Path], output_path: Path | None = None) -> Path:
    """比較多個模型的測試準確率曲線。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for log_path in log_paths:
        df = pd.read_csv(log_path)
        model_name = log_path.stem.replace("_train_log", "")
        test_acc_col = _pick_metric_col(df, preferred="test_acc", fallback="val_acc")
        ax.plot(df["epoch"], df[test_acc_col], label=model_name, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison - Test Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIG_DIR / "compare_test_acc.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_logs(log_paths: list[Path]) -> list[Path]:
    """為每個模型輸出單獨曲線，並輸出一張對比圖。"""
    outputs: list[Path] = []
    for log_path in log_paths:
        outputs.append(plot_single_log(log_path))
    outputs.append(plot_compare_logs(log_paths))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="繪製訓練曲線")
    parser.add_argument("--log", type=str, default=None, help="單一日誌 CSV 路徑")
    parser.add_argument("--compare", action="store_true", help="比較 logs 目錄下所有日誌")
    parser.add_argument("--all", action="store_true", help="輸出所有單模型曲線 + 一張三模型對比圖")
    parser.add_argument("--output", type=str, default=None, help="輸出圖片路徑")
    args = parser.parse_args()

    if args.all:
        log_paths = sorted(LOG_DIR.glob("*_train_log.csv"))
        if not log_paths:
            raise FileNotFoundError(f"在 {LOG_DIR} 中找不到訓練日誌")
        outputs = plot_all_logs(log_paths)
        print("已輸出圖表:")
        for path in outputs:
            print(f"  - {path}")
        return

    if args.compare:
        log_paths = sorted(LOG_DIR.glob("*_train_log.csv"))
        if not log_paths:
            raise FileNotFoundError(f"在 {LOG_DIR} 中找不到訓練日誌")
        out = Path(args.output) if args.output else None
        path = plot_compare_logs(log_paths, out)
        print(f"比較圖已保存: {path}")
        return

    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
    else:
        logs = sorted(LOG_DIR.glob("*_train_log.csv"))
        if not logs:
            raise FileNotFoundError(f"請指定 --log 或在 {LOG_DIR} 中放置訓練日誌")
        log_path = logs[-1]

    out = Path(args.output) if args.output else None
    path = plot_single_log(log_path, out)
    print(f"曲線圖已保存: {path}")


if __name__ == "__main__":
    main()
