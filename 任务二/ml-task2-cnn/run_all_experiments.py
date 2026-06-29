"""一鍵跑通整個 CIFAR-10 CNN 實驗。

功能：
1) 依序訓練 base_cnn / wide_cnn / deep_cnn
2) 評估每個最佳模型並保存文字報告
3) 生成每個模型曲線圖 + 三模型對比圖
4) 匯總核心指標到 CSV
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = OUTPUT_DIR / "models"

DEFAULT_MODELS = ["base_cnn", "wide_cnn", "deep_cnn"]


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """執行命令並即時輸出。"""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"命令執行失敗（exit={result.returncode}）: {' '.join(cmd)}")
    return result


def read_metrics_from_log(log_path: Path) -> dict[str, float | int]:
    """讀取單一模型訓練日誌，提取最終與最佳指標。"""
    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"日誌為空: {log_path}")

    last = rows[-1]
    best_row = max(rows, key=lambda r: float(r["test_acc"]))
    return {
        "epochs": len(rows),
        "final_train_loss": float(last["train_loss"]),
        "final_train_acc": float(last["train_acc"]),
        "final_test_loss": float(last["test_loss"]),
        "final_test_acc": float(last["test_acc"]),
        "best_test_acc": float(best_row["test_acc"]),
        "best_test_loss": float(best_row["test_loss"]),
        "best_epoch": int(best_row["epoch"]),
    }


def write_summary(summary_rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "epochs",
        "final_train_loss",
        "final_train_acc",
        "final_test_loss",
        "final_test_acc",
        "best_test_acc",
        "best_test_loss",
        "best_epoch",
        "log_path",
        "checkpoint_path",
        "eval_report_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="一鍵訓練、評估並生成全部圖表")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="要訓練的模型列表")
    parser.add_argument("--epochs", type=int, default=20, help="訓練輪數")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    parser.add_argument("--device", type=str, default="auto", help="裝置 auto/cuda/mps/cpu")
    parser.add_argument("--skip-train", action="store_true", help="跳過訓練，只做評估與繪圖")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str | float | int]] = []

    for model in args.models:
        model = model.strip()
        if not model:
            continue

        log_path = LOG_DIR / f"{model}_train_log.csv"
        ckpt_path = MODEL_DIR / f"{model}_best.pth"
        eval_report_path = LOG_DIR / f"{model}_eval_report.txt"

        if not args.skip_train:
            train_cmd = [
                sys.executable,
                "train.py",
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--device",
                args.device,
            ]
            run_cmd(train_cmd, cwd=SRC_DIR)

        if not log_path.exists():
            raise FileNotFoundError(f"找不到訓練日誌: {log_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"找不到模型權重: {ckpt_path}")

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            f"outputs/models/{model}_best.pth",
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
        eval_result = run_cmd(eval_cmd, cwd=SRC_DIR)
        eval_report_path.write_text(eval_result.stdout, encoding="utf-8")

        metrics = read_metrics_from_log(log_path)
        summary_rows.append(
            {
                "model": model,
                **metrics,
                "log_path": str(log_path),
                "checkpoint_path": str(ckpt_path),
                "eval_report_path": str(eval_report_path),
            }
        )

    run_cmd([sys.executable, "plot_results.py", "--all"], cwd=SRC_DIR)

    summary_path = LOG_DIR / "experiment_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"\n實驗完成。")
    print(f"- 指標匯總: {summary_path}")
    print(f"- 圖表目錄: {OUTPUT_DIR / 'figures'}")
    print(f"- 模型目錄: {MODEL_DIR}")


if __name__ == "__main__":
    main()
