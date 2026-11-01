"""Low-resource multimodal deep-learning classifier used after local training.

The text branch is a bidirectional GRU; the image branch is a compact CNN; a gated fusion
module combines both learned representations with engineered side features. This architecture
is intentionally lighter than the paper's RoBERTa + ResNet50 stack for CPU-only demo machines.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .features import numeric_feature_vector

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "multimodal_model.pt"
VOCAB_PATH = ARTIFACT_DIR / "vocab.json"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence
except ImportError as exc:  # pragma: no cover - exercised on installations without the DL extra
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


TOKEN_RE = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_TOKENS = 40
IMAGE_SIZE = 96


def tokenize(title: str) -> list[str]:
    return TOKEN_RE.findall(title.lower())[:MAX_TOKENS]


def build_vocab(titles: list[str], max_vocab: int = 12000, min_freq: int = 1) -> dict[str, int]:
    from collections import Counter
    counts = Counter(token for title in titles for token in tokenize(title))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counts.most_common(max(0, max_vocab - 2)):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_title(title: str, vocab: dict[str, int], max_len: int = MAX_TOKENS) -> list[int]:
    ids = [vocab.get(token, vocab.get(UNK_TOKEN, 1)) for token in tokenize(title)[:max_len]]
    ids = ids or [vocab.get(UNK_TOKEN, 1)]
    return ids[:max_len] + [0] * max(0, max_len - len(ids))


def image_to_tensor(image_bytes: bytes | None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install requirements-dl.txt first.")
    if image_bytes:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        except (OSError, ValueError):
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
    else:
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


if torch is not None:
    class MultimodalFusionNet(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64, feature_dim: int = 16):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.text_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.text_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.15)
            )
            self.image_cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, 128), nn.GELU(), nn.Dropout(0.15),
            )
            self.feature_projection = nn.Sequential(nn.Linear(feature_dim, 32), nn.LayerNorm(32), nn.GELU())
            self.fusion_gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
            self.classifier = nn.Sequential(
                nn.Linear(160, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1),
            )

        def forward(self, token_ids, images, numeric_features):
            embedded = self.embedding(token_ids)
            lengths = token_ids.ne(0).sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.text_gru(packed)
            text_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=token_ids.shape[1]
            )
            mask = token_ids.ne(0).unsqueeze(-1)
            masked = text_out.masked_fill(~mask, -1e4)
            max_pool = masked.max(dim=1).values
            mean_pool = (text_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            text_vec = self.text_projection((max_pool + mean_pool) / 2.0)
            image_vec = self.image_cnn(images)
            gate = self.fusion_gate(torch.cat([text_vec, image_vec], dim=1))
            fused = gate * text_vec + (1.0 - gate) * image_vec
            side_vec = self.feature_projection(numeric_features)
            return self.classifier(torch.cat([fused, side_vec], dim=1)).squeeze(1)
else:
    class MultimodalFusionNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for model training and inference.") from _TORCH_IMPORT_ERROR


class TrainedPredictor:
    def __init__(self, model: Any, vocab: dict[str, int], device: str = "cpu"):
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device)
        self.model.eval()

    @classmethod
    def load(cls, model_path: Path = MODEL_PATH, vocab_path: Path = VOCAB_PATH, config_path: Path = CONFIG_PATH):
        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install requirements-dl.txt to load a trained model.")
        for path in (model_path, vocab_path, config_path):
            if not path.exists():
                raise FileNotFoundError(f"Trained artifact missing: {path.name}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MultimodalFusionNet(vocab_size=int(config["vocab_size"]))
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.to(device)
        return cls(model, vocab, device)

    def predict_probability(self, title: str, image_bytes: bytes | None = None) -> float:
        token_ids = torch.tensor([encode_title(title, self.vocab)], dtype=torch.long, device=self.device)
        image = image_to_tensor(image_bytes).unsqueeze(0).to(self.device)
        side = torch.tensor([numeric_feature_vector(title, image_bytes)], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            logit = self.model(token_ids, image, side)
            return float(torch.sigmoid(logit)[0].item())


def trained_artifacts_present() -> bool:
    return MODEL_PATH.exists() and VOCAB_PATH.exists() and CONFIG_PATH.exists()
