import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("lstm_experiment", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入模組：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSTM experiment with isolated output directory")
    parser.add_argument("--run-name", type=str, default=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=2,
        help="連續幾輪驗證 loss 未改善則早停（0 表示關閉）。",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="驗證 loss 最小改善幅度。",
    )
    parser.add_argument(
        "--script-name",
        type=str,
        default="Pytorch_LSTM实战情感分类.py",
        help="Target experiment script filename in current src folder.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main():
    args, passthrough = parse_args()
    src_dir = Path(__file__).resolve().parent
    experiment_root = src_dir.parent
    target_script = src_dir / args.script_name

    module = load_module(target_script)

    run_output = experiment_root / "output" / "experiments" / args.run_name
    module.OUTPUT_DIR = run_output
    module.FIG_DIR = run_output / "figures"
    module.LOG_DIR = run_output / "logs"
    module.MODEL_DIR = run_output / "models"

    print(f"Run name: {args.run_name}")
    print(f"Isolated output directory: {run_output}")

    sys.argv = [
        str(target_script),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        *passthrough,
    ]
    module.main()


if __name__ == "__main__":
    main()
