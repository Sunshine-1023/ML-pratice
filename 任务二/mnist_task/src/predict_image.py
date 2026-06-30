import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from models import build_model
from PIL import Image, ImageOps
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit photo.")
    parser.add_argument("--image", required=True, help="自己拍攝的手寫數字圖片路徑。")
    parser.add_argument("--model", required=True, help="訓練好的 .pth 模型路徑。")
    parser.add_argument(
        "--model-name",
        choices=["simple_cnn", "better_cnn"],
        default=None,
        help="模型檔沒有保存名稱時才需要指定。",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_digit_bbox(gray_image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.array(gray_image)
    light_background = np.median(arr) > 127
    threshold = max(30, min(220, int(arr.mean() - arr.std() * 0.25)))
    mask = arr < threshold if light_background else arr > threshold

    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("找不到明顯的數字區域，請確認照片中有清楚的手寫數字。")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = max(8, int(max(y_max - y_min, x_max - x_min) * 0.15))
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(arr.shape[1] - 1, x_max + pad)
    y_max = min(arr.shape[0] - 1, y_max + pad)
    return x_min, y_min, x_max + 1, y_max + 1


def make_square(image: Image.Image, fill: int) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    square = Image.new("L", (side, side), color=fill)
    offset = ((side - width) // 2, (side - height) // 2)
    square.paste(image, offset)
    return square


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, list[tuple[str, Image.Image]]]:
    original = Image.open(image_path).convert("RGB")
    gray = ImageOps.grayscale(original)
    gray = ImageOps.autocontrast(gray)
    bbox = find_digit_bbox(gray)
    cropped = gray.crop(bbox)

    background_is_light = np.median(np.array(gray)) > 127
    square = make_square(cropped, fill=255 if background_is_light else 0)
    resized = square.resize((28, 28), Image.Resampling.LANCZOS)
    mnist_like = ImageOps.invert(resized) if background_is_light else resized

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    tensor = to_tensor(mnist_like).unsqueeze(0)
    steps = [
        ("原圖", original),
        ("灰階", gray),
        ("裁切數字", cropped),
        ("正方形", square),
        ("28x28 + 反色", mnist_like),
    ]
    return tensor, steps


def save_preprocess_steps(image_path: Path, steps: list[tuple[str, Image.Image]]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"{image_path.stem}_preprocess.png"

    fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 3))
    for ax, (title, image) in zip(axes, steps):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def load_model(model_path: Path, model_name: str | None, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_model_name = checkpoint.get("model_name") if isinstance(checkpoint, dict) else None
    name = model_name or checkpoint_model_name
    if name is None:
        raise ValueError("模型檔沒有保存 model_name，請使用 --model-name 指定模型類型。")

    model = build_model(name).to(device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    device = get_device()

    image_tensor, steps = preprocess_image(image_path)
    steps_path = save_preprocess_steps(image_path, steps)
    model = load_model(model_path, args.model_name, device)

    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    print(f"預測結果：{prediction.item()}")
    print(f"置信度：{confidence.item():.4f}")
    print(f"預處理流程圖已儲存：{steps_path}")


if __name__ == "__main__":
    main()
