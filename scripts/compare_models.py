#!/usr/bin/env python3
"""Train and compare four CPU-friendly deep-learning models on the same data split.

Accepted CSV fields include:
  title/video_title/headline, final_label/label/clickbait/is_clickbait,
  local_thumbnail/thumbnail_path/image_path, and optionally thumbnail URL.

If local thumbnail files are not present, --download-thumbnails downloads only the
stratified subset selected by --max-samples, and only from known YouTube image hosts.
This is intended for a low-resource pilot, not a paper-level replication.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vericlick.deep_model import (  # noqa: E402
    IMAGE_SIZE, MAX_TOKENS, MultimodalFusionNet, build_vocab, encode_title, image_to_tensor,
)
from vericlick.features import numeric_feature_vector  # noqa: E402


class TextBiGRU(nn.Module):
    """Title-only neural baseline: embedding + bidirectional GRU."""
    def __init__(self, vocab_size: int, embedding_dim: int = 48, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim * 4, 48), nn.GELU(), nn.Dropout(0.15), nn.Linear(48, 1))

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        output, _ = self.gru(embedded)
        mask = token_ids.ne(0).unsqueeze(-1)
        masked = output.masked_fill(~mask, -1e4)
        max_pool = masked.max(dim=1).values
        mean_pool = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(torch.cat([max_pool, mean_pool], dim=1)).squeeze(1)


class ThumbnailCNN(nn.Module):
    """Thumbnail-only neural baseline with a compact CNN."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1))

    def forward(self, images):
        return self.head(self.features(images)).squeeze(1)


class FeatureMLP(nn.Module):
    """Engineered-feature neural baseline (16 lexical and thumbnail-style features)."""
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(1)


class LabeledVideoDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: dict[str, int]):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row["thumbnail"]))
        image_bytes = path.read_bytes()
        return {
            "token_ids": torch.tensor(encode_title(str(row["video_title"]), self.vocab, MAX_TOKENS), dtype=torch.long),
            "image": image_to_tensor(image_bytes),
            "features": torch.tensor(numeric_feature_vector(str(row["video_title"]), image_bytes), dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Labeled dataset CSV, e.g. data/Final_data.csv")
    parser.add_argument("--image-root", default=None, help="Folder containing extracted local thumbnails; defaults to CSV parent")
    parser.add_argument("--download-thumbnails", action="store_true", help="Download only sampled missing thumbnails from allow-listed YouTube image hosts")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=300, help="Stratified pilot size; keep small on CPU-only laptops")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads used by PyTorch")
    parser.add_argument("--artifact-dir", default=str(ROOT / "artifacts"), help="Output folder; defaults to project artifacts/")
    return parser.parse_args()


def _column(lookup: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((lookup[name] for name in names if name in lookup), None)


def normalize_raw_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize the public YTClickbait21K schema and common custom CSV schemas."""
    lookup = {str(column).strip().lower(): column for column in raw.columns}
    title_col = _column(lookup, ("video_title", "title", "headline", "video title"))
    label_col = _column(lookup, ("final_label", "label", "clickbait", "target", "is_clickbait"))
    local_col = _column(lookup, ("local_thumbnail", "thumbnail_path", "image_path", "image_file", "local_image"))
    thumb_col = _column(lookup, ("thumbnail", "thumbnail_url", "image_url"))
    video_id_col = _column(lookup, ("video_id", "id"))
    missing = [name for name, col in (("title/video_title", title_col), ("label/final_label", label_col), ("local_thumbnail or thumbnail URL", local_col or thumb_col)) if col is None]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}. Columns found: {list(raw.columns)}")
    frame = pd.DataFrame({
        "video_title": raw[title_col].fillna("").astype(str).str.strip(),
        "label": pd.to_numeric(raw[label_col], errors="coerce"),
        "local_thumbnail": (raw[local_col] if local_col else raw[thumb_col] if thumb_col else "").fillna("").astype(str).str.strip() if (local_col or thumb_col) else "",
        "thumbnail_url": raw[thumb_col].fillna("").astype(str).str.strip() if thumb_col else "",
        "video_id": raw[video_id_col].fillna("").astype(str).str.strip() if video_id_col else "",
    })
    frame = frame[(frame.video_title.str.len() >= 3) & frame.label.isin([0, 1])].copy()
    frame["label"] = frame["label"].astype(int)
    frame = frame.drop_duplicates(subset=["video_title", "label", "video_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid labeled rows were found. Labels must be 0 or 1 and titles must be non-empty.")
    if frame.label.nunique() < 2:
        raise ValueError("The selected data must contain both label classes: 0=non-clickbait and 1=clickbait.")
    return frame


def _valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        allowed = host in {"i.ytimg.com", "img.youtube.com"} or host.endswith(".ytimg.com")
        return parsed.scheme == "https" and allowed
    except ValueError:
        return False


def _resolve_local(value: str, image_root: Path, csv_parent: Path, basename_map: dict[str, Path]) -> Path | None:
    value = str(value or "").strip()
    if not value or value.lower() in {"nan", "none", "null"} or value.startswith(("http://", "https://")):
        return None
    raw_path = Path(value)
    candidates = [raw_path] if raw_path.is_absolute() else [image_root / raw_path, csv_parent / raw_path]
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size <= 5 * 1024 * 1024:
                return candidate.resolve()
        except OSError:
            continue
    return basename_map.get(raw_path.name.lower())


def _download_image(url: str, out_path: Path) -> Path | None:
    if not _valid_url(url):
        return None
    try:
        import requests
        response = requests.get(url, timeout=(4, 10), headers={"User-Agent": "Veri-Click-research/1.0"}, stream=True, allow_redirects=False)
        if response.status_code != 200 or "image" not in response.headers.get("content-type", "").lower():
            response.close()
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with out_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=32768):
                if not chunk:
                    continue
                total += len(chunk)
                if total > 5 * 1024 * 1024:
                    handle.close()
                    out_path.unlink(missing_ok=True)
                    response.close()
                    return None
                handle.write(chunk)
        response.close()
        with Image.open(out_path) as im:
            im.verify()
        return out_path.resolve()
    except Exception:
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def resolve_thumbnails(frame: pd.DataFrame, image_root: Path, csv_parent: Path, download: bool) -> pd.DataFrame:
    image_root.mkdir(parents=True, exist_ok=True)
    # A basename index also handles archives that include a top-level folder prefix.
    basename_map: dict[str, Path] = {}
    if image_root.exists():
        for path in image_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                basename_map.setdefault(path.name.lower(), path.resolve())
    resolved = []
    failures = 0
    total_rows = len(frame)
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        path = _resolve_local(row.local_thumbnail, image_root, csv_parent, basename_map)
        if path is None and download and _valid_url(row.thumbnail_url):
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", row.video_id or Path(urlparse(row.thumbnail_url).path).stem or "thumbnail")[:80]
            suffix = Path(urlparse(row.thumbnail_url).path).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".jpg"
            path = _download_image(row.thumbnail_url, image_root / "downloaded_thumbnails" / f"{stem}{suffix}")
            if path:
                basename_map[path.name.lower()] = path
        resolved.append(str(path) if path else "")
        if path is None:
            failures += 1
        if download and (index % 25 == 0 or index == total_rows):
            print(f"Thumbnail preparation: {index}/{total_rows} rows checked")
    output = frame.copy()
    output["thumbnail"] = resolved
    output = output[output.thumbnail.str.len() > 0].reset_index(drop=True)
    print(f"Resolved local thumbnails: {len(output)}; missing/unusable: {failures}")
    return output


def metrics(labels, probabilities) -> dict:
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], target_names=["non_clickbait", "clickbait"], output_dict=True, zero_division=0),
    }


def forward(model, kind: str, batch, device):
    tokens = batch["token_ids"].to(device)
    images = batch["image"].to(device)
    features = batch["features"].to(device)
    if kind == "bigru_text":
        return model(tokens)
    if kind == "cnn_image":
        return model(images)
    if kind == "mlp_features":
        return model(features)
    return model(tokens, images, features)


def run_epoch(model, kind, loader, optimizer, criterion, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_labels, all_probs = [], []
    for batch in loader:
        labels = batch["label"].to(device)
        with torch.set_grad_enabled(training):
            logits = forward(model, kind, batch, device)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return total_loss / max(len(loader.dataset), 1), metrics(all_labels, all_probs), all_labels, all_probs


def _make_model(kind: str, vocab_size: int):
    if kind == "bigru_text":
        return TextBiGRU(vocab_size)
    if kind == "cnn_image":
        return ThumbnailCNN()
    if kind == "mlp_features":
        return FeatureMLP()
    return MultimodalFusionNet(vocab_size=vocab_size)


def train_one(name, kind, train_loader, valid_loader, test_loader, vocab_size, train_labels, args, device, comp_dir):
    model = _make_model(kind, vocab_size).to(device)
    positive = max(int(np.sum(np.asarray(train_labels) == 1)), 1)
    negative = max(int(np.sum(np.asarray(train_labels) == 0)), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_val_f1 = -1.0
    stale = 0
    best_epoch = 0
    best_path = comp_dir / f"{kind}.pt"
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(model, kind, train_loader, optimizer, criterion, device, True)
        valid_loss, valid_metrics, _, _ = run_epoch(model, kind, valid_loader, optimizer, criterion, device, False)
        history.append({"model": kind, "epoch": epoch, "train_loss": round(train_loss, 5), "train_f1": train_metrics["f1"], "valid_loss": round(valid_loss, 5), "valid_f1": valid_metrics["f1"]})
        print(f"[{name}] epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, F1 {train_metrics['f1']:.3f} | val loss {valid_loss:.4f}, F1 {valid_metrics['f1']:.3f}")
        if valid_metrics["f1"] > best_val_f1:
            best_val_f1 = valid_metrics["f1"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[{name}] early stopping after {args.patience} non-improving epoch(s)")
                break
    try:
        state = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    _, test_metrics, _, _ = run_epoch(model, kind, test_loader, optimizer, criterion, device, False)
    elapsed = round(time.time() - start, 1)
    print(f"[{name}] held-out test | accuracy {test_metrics['accuracy']:.3f}, precision {test_metrics['precision']:.3f}, recall {test_metrics['recall']:.3f}, F1 {test_metrics['f1']:.3f}")
    return model, {"key": kind, "name": name, "status": "trained", "best_epoch": best_epoch, "training_seconds": elapsed, "validation_f1_at_best_epoch": round(best_val_f1, 4), "test_metrics": test_metrics}, history


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.threads < 1:
        raise SystemExit("--epochs, --patience, and --threads must be positive; --batch-size must be at least 2.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file(): raise SystemExit(f"Dataset CSV not found: {csv_path}")
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else csv_path.parent
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    comp_dir = artifact_dir / "comparison_models"
    comp_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    frame = normalize_raw_dataset(raw)
    if args.max_samples > 0 and len(frame) > args.max_samples:
        frame, _ = train_test_split(frame, train_size=args.max_samples, random_state=args.seed, stratify=frame["label"])
        frame = frame.reset_index(drop=True)
        print(f"Using a stratified pilot sample of {len(frame)} rows from {len(raw)} CSV rows.")
    elif args.max_samples <= 0:
        print(f"No sample cap: attempting all {len(frame)} labeled rows. This may be slow on CPU.")

    frame = resolve_thumbnails(frame, image_root, csv_path.parent, args.download_thumbnails)
    counts = frame.label.value_counts().to_dict()
    if len(frame) < 40 or frame.label.nunique() < 2 or frame.label.value_counts().min() < 10:
        raise SystemExit(
            f"Not enough usable title+thumbnail rows. Need >=40 rows and >=10 per class after matching local thumbnails. "
            f"Valid rows: {len(frame)}; class counts: {counts}. Extract thumbnails or retry with --download-thumbnails and a small --max-samples."
        )

    train_df, holdout = train_test_split(frame, test_size=0.30, random_state=args.seed, stratify=frame.label)
    valid_df, test_df = train_test_split(holdout, test_size=0.50, random_state=args.seed, stratify=holdout.label)
    vocab = build_vocab(train_df.video_title.tolist())  # built from train split only
    train_ds, valid_ds, test_ds = LabeledVideoDataset(train_df, vocab), LabeledVideoDataset(valid_df, vocab), LabeledVideoDataset(test_df, vocab)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_labels = train_df.label.tolist()

    model_specs = [
        ("BiGRU text-only", "bigru_text"),
        ("CNN thumbnail-only", "cnn_image"),
        ("MLP engineered features", "mlp_features"),
        ("Multimodal gated fusion", "multimodal_fusion"),
    ]
    all_results, all_history = [], []
    fusion_model = None
    overall_start = time.time()
    for name, kind in model_specs:
        model, result, history = train_one(name, kind, train_loader, valid_loader, test_loader, len(vocab), train_labels, args, device, comp_dir)
        all_results.append(result); all_history.extend(history)
        if kind == "multimodal_fusion": fusion_model = model

    if fusion_model is None: raise RuntimeError("Fusion model did not train.")
    # Deploy the primary fusion model using the application's existing artifact contract.
    torch.save(fusion_model.state_dict(), artifact_dir / "multimodal_model.pt")
    (artifact_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    fusion_result = next(item for item in all_results if item["key"] == "multimodal_fusion")
    config = {
        "model_name": "Veri-Click compact multimodal BiGRU-CNN gated fusion",
        "vocab_size": len(vocab), "max_tokens": MAX_TOKENS, "image_size": IMAGE_SIZE,
        "feature_dim": 16, "seed": args.seed, "best_epoch": fusion_result["best_epoch"],
        "train_rows": len(train_df), "validation_rows": len(valid_df), "test_rows": len(test_df),
        "dataset_csv": str(csv_path), "image_root": str(image_root), "training_device": str(device),
        "model_comparison": True, "elapsed_seconds": round(time.time() - overall_start, 1),
        "note": "Compact alternative architecture; these dataset-specific results do not reproduce the supplied paper's RoBERTa/ResNet50 stacked-ensemble benchmark.",
    }
    (artifact_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (artifact_dir / "test_metrics.json").write_text(json.dumps({"config": config, "test": fusion_result["test_metrics"], "history": [row for row in all_history if row["model"] == "multimodal_fusion"]}, indent=2), encoding="utf-8")

    comparison = {
        "status": "completed", "dataset_file": csv_path.name, "input_rows": int(len(raw)), "valid_rows": int(len(frame)),
        "class_counts": {str(k): int(v) for k, v in frame.label.value_counts().to_dict().items()},
        "split": {"train": int(len(train_df)), "validation": int(len(valid_df)), "test": int(len(test_df)), "seed": args.seed},
        "device": str(device), "epochs_requested": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "patience": args.patience, "max_samples_argument": args.max_samples,
        "elapsed_seconds": round(time.time() - overall_start, 1), "primary_model": "multimodal_fusion",
        "models": all_results,
        "interpretation": "All models use the same stratified train/validation/test rows. Report held-out test metrics, not training metrics. Small pilot scores are noisy and do not reproduce the supplied paper.",
    }
    (artifact_dir / "model_comparison_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    with (artifact_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_seconds", "best_epoch"])
        writer.writeheader()
        for result in all_results:
            met = result["test_metrics"]
            writer.writerow({"model": result["name"], "accuracy": met["accuracy"], "precision": met["precision"], "recall": met["recall"], "f1": met["f1"], "roc_auc": met["roc_auc"], "train_seconds": result["training_seconds"], "best_epoch": result["best_epoch"]})
    with (artifact_dir / "model_comparison_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss", "train_f1", "valid_loss", "valid_f1"])
        writer.writeheader(); writer.writerows(all_history)
    # Save the exact image-backed rows used, making the experiment auditable/re-runnable.
    frame[["video_title", "label", "thumbnail"]].to_csv(comp_dir / "prepared_dataset_used.csv", index=False)

    print("\nHELD-OUT TEST COMPARISON (same split for every model)")
    print(f"{'Model':28s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s} {'AUC':>7s}")
    for result in all_results:
        m = result["test_metrics"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        print(f"{result['name'][:28]:28s} {m['accuracy']:7.3f} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {auc:>7s}")
    print(f"\nSaved comparison: {artifact_dir / 'model_comparison_metrics.json'}")
    print(f"Saved comparison CSV: {artifact_dir / 'model_comparison.csv'}")
    print(f"Primary fusion checkpoint: {artifact_dir / 'multimodal_model.pt'}")
    print("Restart the Veri-Click API to load the trained fusion checkpoint.")


if __name__ == "__main__":
    main()
