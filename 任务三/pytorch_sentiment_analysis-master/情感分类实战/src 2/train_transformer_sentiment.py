import argparse
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"


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


def ensure_output_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_labels(label_series: pd.Series) -> pd.Series:
    labels = pd.to_numeric(label_series, errors="coerce").dropna().astype(int)
    unique_values = set(labels.unique().tolist())

    if unique_values.issubset({0, 1}):
        return labels
    return labels.map({0: 0, 4: 1})


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


class SentimentTransformerDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def evaluate(model, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            total_steps += 1

            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().tolist())

    loss = total_loss / max(total_steps, 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return loss, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Sentiment140 Transformer fine-tuning")
    parser.add_argument("--model-name", type=str, default="cardiffnlp/twitter-roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--processed-file", type=str, default=str(DATA_DIR / "train-processed.csv"))
    parser.add_argument("--raw-file", type=str, default=str(DATA_DIR / "training.1600000.processed.noemoticon.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_output_dirs()
    device = detect_device()
    print(f"Using device: {device}")

    texts, labels = load_text_label_data(
        processed_file=Path(args.processed_file),
        raw_file=Path(args.raw_file),
        max_samples=args.max_samples,
        seed=args.seed,
    )

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=args.val_size + args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    val_ratio = args.val_size / (args.val_size + args.test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=1 - val_ratio,
        random_state=args.seed,
        stratify=temp_labels,
    )
    print(f"Train/Val/Test = {len(train_texts)}/{len(val_texts)}/{len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    train_ds = SentimentTransformerDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_ds = SentimentTransformerDataset(val_texts, val_labels, tokenizer, args.max_len)
    test_ds = SentimentTransformerDataset(test_texts, test_labels, tokenizer, args.max_len)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.num_workers,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    update_steps_per_epoch = max((len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps, 1)
    total_update_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    model_save_dir = MODEL_DIR / "transformer_best"
    best_val_acc = 0.0
    no_improve_epochs = 0
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", dynamic_ncols=True)
        for step, batch in enumerate(train_bar, start=1):
            labels = batch["labels"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            total_train_loss += outputs.loss.item()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | "
            f"test_acc={test_acc:.4f} test_f1={test_f1:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)
            print(f"Saved best model to: {model_save_dir}")
        else:
            no_improve_epochs += 1
            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                print(f"Early stop triggered at epoch {epoch}.")
                break

    log_file = LOG_DIR / "transformer_training_history.csv"
    pd.DataFrame(history_rows).to_csv(log_file, index=False)
    print(f"Training history saved to: {log_file}")
    print(f"Best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
