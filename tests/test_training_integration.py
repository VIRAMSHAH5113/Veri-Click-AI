from __future__ import annotations

import importlib.util
import json
import sys

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Optional PyTorch dependency is needed for training integration")


def test_tiny_training_to_saved_checkpoint_to_inference(tmp_path, monkeypatch):
    """Integration smoke test: CSV + image files -> training -> held-out metrics -> predictor."""
    import pandas as pd
    import torch
    from PIL import Image, ImageDraw

    from scripts import train_model
    from vericlick.deep_model import TrainedPredictor

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        image_root = tmp_path / "thumbs"
        image_root.mkdir()
        rows = []
        for index in range(40):
            is_clickbait = index % 2
            image = Image.new("RGB", (48, 32), (220, 60, 70) if is_clickbait else (40, 140, 190))
            draw = ImageDraw.Draw(image)
            draw.rectangle((index % 20, 3, 24 + index % 20, 25), fill=(245, 220, 40) if is_clickbait else (230, 245, 250))
            rel_path = f"thumbs/thumb_{index}.png"
            image.save(tmp_path / rel_path)
            title = (
                f"STOP shocking secret reveal urgent example number {index}"
                if is_clickbait else
                f"Beginner tutorial step by step learning example number {index}"
            )
            rows.append({"video_title": title, "label": is_clickbait, "thumbnail": rel_path})
        csv_path = tmp_path / "tiny_dataset.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        artifacts = tmp_path / "trained-artifacts"

        monkeypatch.setattr(sys, "argv", [
            "train_model.py", "--csv", str(csv_path), "--image-root", str(tmp_path),
            "--epochs", "1", "--batch-size", "32", "--patience", "1", "--seed", "7",
            "--artifact-dir", str(artifacts),
        ])
        train_model.main()

        expected = ["multimodal_model.pt", "vocab.json", "model_config.json", "test_metrics.json", "training_history.csv"]
        assert all((artifacts / name).is_file() for name in expected)
        metrics = json.loads((artifacts / "test_metrics.json").read_text(encoding="utf-8"))
        assert metrics["config"]["test_rows"] > 0
        assert 0.0 <= metrics["test"]["accuracy"] <= 1.0

        predictor = TrainedPredictor.load(
            model_path=artifacts / "multimodal_model.pt",
            vocab_path=artifacts / "vocab.json",
            config_path=artifacts / "model_config.json",
        )
        probability = predictor.predict_probability("A step by step tutorial", None)
        assert 0.0 <= probability <= 1.0
    finally:
        torch.set_num_threads(old_threads)
