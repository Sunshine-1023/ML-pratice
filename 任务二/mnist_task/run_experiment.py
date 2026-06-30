import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MNIST experiment.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for each model.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and only plot/predict from existing outputs.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip drawing comparison figures.",
    )
    parser.add_argument(
        "--predict-image",
        type=Path,
        default=None,
        help="Optional image path to predict after training, for example images/digit_7.jpg.",
    )
    parser.add_argument(
        "--predict-model",
        choices=["simple_cnn", "better_cnn"],
        default="better_cnn",
        help="Which trained model to use for prediction.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def train_model(model_name: str, args: argparse.Namespace) -> None:
    run_command(
        [
            sys.executable,
            "src/train.py",
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
        ]
    )


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        train_model("simple_cnn", args)
        train_model("better_cnn", args)

    if not args.skip_plot:
        run_command([sys.executable, "src/plot_results.py"])

    if args.predict_image is not None:
        model_path = PROJECT_ROOT / "outputs" / "models" / f"{args.predict_model}.pth"
        run_command(
            [
                sys.executable,
                "src/predict_image.py",
                "--image",
                str(args.predict_image),
                "--model",
                str(model_path),
            ]
        )

    print("\n實驗執行完成。")


if __name__ == "__main__":
    main()
