#!/usr/bin/env python3
"""Train Veri-Click's compact multimodal deep-learning model from a labeled CSV.

Expected columns: video_title, label (1=clickbait, 0=non-clickbait), thumbnail (image path).
Paths in `thumbnail` are resolved relative to --image-root (defaults to the CSV folder).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericlick.deep_model import (  # noqa: E402
    ARTIFACT_DIR, CONFIG_PATH, IMAGE_SIZE, MAX_TOKENS, MODEL_PATH, VOCAB_PATH,
    MultimodalFusionNet, build_vocab, encode_title, image_to_tensor,
)
from vericlick.features import numeric_feature_vector  # noqa: E402


class VideoDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int], image_root: Path):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.image_root = image_root

    def __len__(self):
        return len(self.frame)

    def _image_bytes(self, path_value) -> bytes | None:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = self.image_root / path
        try:
            data = path.read_bytes()
            return data if len(data) <= 5 * 1024 * 1024 else None
        except OSError:
            return None

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        title = str(row["video_title"])
        image_bytes = self._image_bytes(row["thumbnail"])
        return {
            "token_ids": torch.tensor(encode_title(title, self.vocab, MAX_TOKENS), dtype=torch.long),
            "image": image_to_tensor(image_bytes),
            "features": torch.tensor(numeric_feature_vector(title, image_bytes), dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to labeled dataset CSV")
    parser.add_argument("--image-root", default=None, help="Root folder for thumbnail paths; defaults to CSV parent")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional stratified-size cap; 0 uses all valid rows")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", default=str(ARTIFACT_DIR), help="Output directory for model weights and evaluation artifacts")
    return parser.parse_args()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    title_col = next((lookup[c] for c in ("video_title", "title", "video title") if c in lookup), None)
    label_col = next((lookup[c] for c in ("label", "final_label", "clickbait", "target", "is_clickbait") if c in lookup), None)
    thumb_col = next((lookup[c] for c in ("local_thumbnail", "thumbnail_path", "image_path", "image", "thumbnail") if c in lookup), None)
    missing = [name for name, col in (("video_title", title_col), ("label", label_col), ("thumbnail", thumb_col)) if col is None]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}. Found: {list(frame.columns)}")
    result = frame[[title_col, label_col, thumb_col]].copy()
    result.columns = ["video_title", "label", "thumbnail"]
    result["video_title"] = result["video_title"].fillna("").astype(str).str.strip()
    result["label"] = pd.to_numeric(result["label"], errors="coerce")
    result["thumbnail"] = result["thumbnail"].fillna("").astype(str).str.strip()
    result = result[(result["video_title"].str.len() >= 3) & result["label"].isin([0, 1]) & (result["thumbnail"].str.len() > 0)]
    result["label"] = result["label"].astype(int)
    return result.drop_duplicates(subset=["video_title", "thumbnail"]).reset_index(drop=True)


def _metrics(labels, probs) -> dict:
    preds = (np.asarray(probs) >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = None
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, preds, labels=[0, 1], target_names=["non_clickbait", "clickbait"], output_dict=True, zero_division=0),
    }


def run_epoch(model, loader, optimizer, criterion, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_labels, all_probs = [], []
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        images = batch["image"].to(device)
        features = batch["features"].to(device)
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(token_ids, images, features)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return total_loss / max(len(loader.dataset), 1), _metrics(all_labels, all_probs), all_labels, all_probs


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive integers.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    model_path = artifact_dir / "multimodal_model.pt"
    vocab_path = artifact_dir / "vocab.json"
    config_path = artifact_dir / "model_config.json"
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"Dataset CSV not found: {csv_path}")
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else csv_path.parent
    raw = pd.read_csv(csv_path)
    frame = normalize_columns(raw)
    frame["_resolved"] = frame["thumbnail"].map(lambda p: str((Path(p) if Path(p).is_absolute() else image_root / p).resolve()))
    frame = frame[frame["_resolved"].map(lambda p: Path(p).is_file())].copy()
    frame["thumbnail"] = frame["_resolved"]
    frame.drop(columns=["_resolved"], inplace=True)
    if args.max_samples and args.max_samples > 0 and len(frame) > args.max_samples:
        frame, _ = train_test_split(frame, train_size=args.max_samples, random_state=args.seed, stratify=frame["label"])
        frame = frame.reset_index(drop=True)
    if len(frame) < 40 or frame["label"].nunique() < 2 or frame["label"].value_counts().min() < 10:
        raise SystemExit(
            f"Need at least 40 valid rows and 10 samples per class with existing thumbnail files. Valid rows: {len(frame)}; class counts: {frame['label'].value_counts().to_dict()}"
        )
    train_df, holdout = train_test_split(frame, test_size=0.30, random_state=args.seed, stratify=frame["label"])
    valid_df, test_df = train_test_split(holdout, test_size=0.50, random_state=args.seed, stratify=holdout["label"])
    vocab = build_vocab(train_df["video_title"].tolist())
    train_ds = VideoDataset(train_df, vocab, image_root)
    valid_ds = VideoDataset(valid_df, vocab, image_root)
    test_ds = VideoDataset(test_df, vocab, image_root)
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFusionNet(vocab_size=len(vocab)).to(device)
    pos_count = int((train_df["label"] == 1).sum())
    neg_count = int((train_df["label"] == 0).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, True)
        valid_loss, valid_metrics, _, _ = run_epoch(model, valid_loader, optimizer, criterion, device, False)
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "train_f1": train_metrics["f1"], "valid_loss": round(valid_loss, 5), "valid_f1": valid_metrics["f1"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, F1 {train_metrics['f1']:.3f} | val loss {valid_loss:.4f}, F1 {valid_metrics['f1']:.3f}")
        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {args.patience} non-improving epoch(s).")
                break

    # Reload the selected validation checkpoint before reporting held-out test results.
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch compatibility
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    _, test_metrics, test_labels, test_probs = run_epoch(model, test_loader, optimizer, criterion, device, False)
    vocab_out = json.dumps(vocab, ensure_ascii=False, indent=2)
    vocab_path.write_text(vocab_out, encoding="utf-8")
    config = {
        "model_name": "Veri-Click compact multimodal BiGRU-CNN gated fusion",
        "vocab_size": len(vocab), "max_tokens": MAX_TOKENS, "image_size": IMAGE_SIZE,
        "feature_dim": 16, "seed": args.seed, "best_epoch": best_epoch,
        "train_rows": len(train_df), "validation_rows": len(valid_df), "test_rows": len(test_df),
        "image_root": str(image_root), "dataset_csv": str(csv_path),
        "training_device": str(device), "elapsed_seconds": round(time.time() - start, 1),
        "note": "These are results for this dataset/split only; not directly comparable to figures in the supplied paper.",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path = artifact_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps({"config": config, "test": test_metrics, "history": history}, indent=2), encoding="utf-8")
    with (artifact_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    print("\nHeld-out TEST metrics (not training metrics):")
    print(json.dumps({k: v for k, v in test_metrics.items() if k not in ("classification_report",)}, indent=2))
    print(f"\nSaved model: {model_path}")
    print(f"Saved model metadata: {config_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
