import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_NAMES = ["simple_cnn", "better_cnn"]


def read_history(model_name: str) -> list[dict[str, float]]:
    log_path = LOG_DIR / f"{model_name}_history.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"找不到訓練紀錄：{log_path}")

    with log_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            {
                "epoch": float(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_loss": float(row["test_loss"]),
                "test_acc": float(row["test_acc"]),
            }
            for row in reader
        ]


def plot_metric(
    histories: dict[str, list[dict[str, float]]],
    metric: str,
    ylabel: str,
    output_name: str,
) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / output_name

    plt.figure(figsize=(8, 5))
    for model_name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        values = [row[metric] for row in history]
        plt.plot(epochs, values, marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_final_accuracy(histories: dict[str, list[dict[str, float]]]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / "final_test_accuracy_compare.png"

    model_names = list(histories.keys())
    accuracies = [histories[name][-1]["test_acc"] for name in model_names]

    plt.figure(figsize=(6.4, 5.2))
    bars = plt.bar(model_names, accuracies, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1.04)
    plt.ylabel("Test Accuracy")
    plt.title("Final Test Accuracy Comparison", pad=14)
    for bar, acc in zip(bars, accuracies):
        text_y = min(acc + 0.004, 1.018)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{acc:.2%}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main() -> None:
    histories = {model_name: read_history(model_name) for model_name in MODEL_NAMES}
    saved_files = [
        plot_metric(histories, "train_loss", "Train Loss", "train_loss_compare.png"),
        plot_metric(histories, "test_loss", "Test Loss", "test_loss_compare.png"),
        plot_metric(histories, "train_acc", "Train Accuracy", "train_accuracy_compare.png"),
        plot_metric(histories, "test_acc", "Test Accuracy", "test_accuracy_compare.png"),
        plot_final_accuracy(histories),
    ]

    print("圖表已儲存：")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    main()
