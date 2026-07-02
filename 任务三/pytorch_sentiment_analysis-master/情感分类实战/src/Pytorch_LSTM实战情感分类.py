import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
FIG_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = OUTPUT_DIR / "models"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_env_info(device: torch.device) -> None:
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"Torch MPS available: {mps_ok}")
    print(f"Using device: {device}")


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def normalize_labels(label_series: pd.Series) -> pd.Series:
    labels = pd.to_numeric(label_series, errors="coerce").dropna().astype(int)
    unique_values = set(labels.unique().tolist())

    if unique_values.issubset({0, 1}):
        return labels

    mapped = labels.map({0: 0, 4: 1})
    return mapped


def load_text_label_data(
    processed_file: Path,
    raw_file: Path,
    max_samples: int,
    seed: int,
) -> Tuple[List[str], List[int]]:
    if processed_file.exists():
        df = pd.read_csv(
            processed_file,
            header=None,
            encoding="ISO-8859-1",
            engine="python",
            on_bad_lines="skip",
        )
        if df.shape[1] < 8:
            raise ValueError("train-processed.csv 欄位不足，至少需要 8 欄。")
        text_col = df.iloc[:, 5]
        label_col = normalize_labels(df.iloc[:, 7])
        df = pd.DataFrame({"text": text_col, "label": label_col})
    elif raw_file.exists():
        df = pd.read_csv(
            raw_file,
            header=None,
            encoding="ISO-8859-1",
            engine="python",
            on_bad_lines="skip",
        )
        if df.shape[1] < 6:
            raise ValueError("原始資料欄位不足，至少需要 6 欄。")
        text_col = df.iloc[:, 5]
        label_col = normalize_labels(df.iloc[:, 0])
        df = pd.DataFrame({"text": text_col, "label": label_col})
    else:
        raise FileNotFoundError("找不到 train-processed.csv 或原始訓練資料。")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    print(f"Loaded samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")
    return df["text"].tolist(), df["label"].tolist()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(texts: List[str], max_vocab_size: int, min_freq: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    stoi: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(stoi) >= max_vocab_size:
            break
        stoi[token] = len(stoi)
    print(f"Vocab size: {len(stoi)}")
    return stoi


class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], stoi: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len
        self.unk_idx = stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        tokens = tokenize(self.texts[idx])[: self.max_len]
        token_ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        if not token_ids:
            token_ids = [self.unk_idx]
        return (
            torch.tensor(token_ids, dtype=torch.long),
            len(token_ids),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def make_collate_fn(pad_idx: int):
    def collate_fn(batch):
        sequences, lengths, labels = zip(*batch)
        padded = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
        labels_tensor = torch.stack(labels)
        return padded, lengths_tensor, labels_tensor

    return collate_fn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.predictor = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embeddings,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        hidden = self.dropout(hidden[-1])
        return self.predictor(hidden)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_mode: bool,
    epoch_idx: int,
    total_epochs: int,
    stage: str,
) -> Tuple[float, float]:
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        dataloader,
        desc=f"Epoch {epoch_idx:02d}/{total_epochs:02d} [{stage}]",
        leave=False,
        dynamic_ncols=True,
    )
    for input_ids, lengths, labels in progress:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_samples, 1):.4f}",
        )

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def save_curves(history: Dict[str, List[float]]) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lstm_loss.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.plot(epochs, history["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lstm_accuracy.png", dpi=150)
    plt.close()

    history_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "test_loss": history["test_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
            "test_acc": history["test_acc"],
        }
    )
    history_df.to_csv(LOG_DIR / "training_history.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment140 LSTM (PyTorch 2.x, no torchtext)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dirs()

    device = detect_device()
    print_env_info(device)

    processed_file = DATA_DIR / "train-processed.csv"
    raw_file = DATA_DIR / "training.1600000.processed.noemoticon.csv"

    texts, labels = load_text_label_data(
        processed_file=processed_file,
        raw_file=raw_file,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    temp_size = args.val_size + args.test_size
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=temp_size,
        random_state=args.seed,
        stratify=labels,
    )
    test_ratio_in_temp = args.test_size / temp_size
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=test_ratio_in_temp,
        random_state=args.seed,
        stratify=temp_labels,
    )

    print(f"Train / Val / Test = {len(train_texts)} / {len(val_texts)} / {len(test_texts)}")

    stoi = build_vocab(train_texts, args.max_vocab_size, args.min_freq)
    train_ds = SentimentDataset(train_texts, train_labels, stoi, args.max_len)
    val_ds = SentimentDataset(val_texts, val_labels, stoi, args.max_len)
    test_ds = SentimentDataset(test_texts, test_labels, stoi, args.max_len)

    collate_fn = make_collate_fn(stoi[PAD_TOKEN])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = LSTMClassifier(
        vocab_size=len(stoi),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }

    best_val_loss = float("inf")
    best_model_path = MODEL_DIR / "best_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            stage="train",
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            stage="val",
        )
        test_loss, test_acc = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            stage="test",
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    save_curves(history)
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {LOG_DIR / 'training_history.csv'}")


if __name__ == "__main__":
    main()
