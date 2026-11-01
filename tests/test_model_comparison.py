from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is required for model-comparison tests")


def test_public_dataset_schema_is_normalized():
    from scripts.compare_models import normalize_raw_dataset

    frame = normalize_raw_dataset(pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "title": ["Sensational title one", "Helpful tutorial two", "A third example title"],
        "final_label": [1, 0, 1],
        "thumbnail": ["https://i.ytimg.com/vi/a/hqdefault.jpg", "https://i.ytimg.com/vi/b/hqdefault.jpg", "https://i.ytimg.com/vi/c/hqdefault.jpg"],
        "local_thumbnail": ["", "", ""],
    }))
    assert list(frame["label"]) == [1, 0, 1]
    assert list(frame["video_title"])[0] == "Sensational title one"
    assert frame.loc[0, "thumbnail_url"].startswith("https://i.ytimg.com/")


def test_thumbnail_download_hosts_are_allow_listed():
    from scripts.compare_models import _valid_url

    assert _valid_url("https://i.ytimg.com/vi/abc/hqdefault.jpg")
    assert not _valid_url("http://i.ytimg.com/vi/abc/hqdefault.jpg")
    assert not _valid_url("https://example.com/thumb.jpg")


def test_four_model_comparison_smoke_training(tmp_path, monkeypatch):
    """Tiny synthetic fixture tests the pipeline contract; these scores are not research results."""
    import json
    import sys
    from PIL import Image, ImageDraw
    import torch
    from scripts import compare_models
    from vericlick.deep_model import TrainedPredictor

    image_root = tmp_path / "thumbs"
    image_root.mkdir()
    rows = []
    for index in range(40):
        positive = index % 2
        image = Image.new("RGB", (64, 48), (225, 60, 70) if positive else (35, 135, 195))
        ImageDraw.Draw(image).rectangle((5, 5, 35, 35), fill=(250, 220, 30) if positive else (230, 245, 250))
        path = image_root / f"thumb_{index}.jpg"
        image.save(path)
        rows.append({
            "video_id": f"video-{index}",
            "title": (f"STOP shocking unbelievable secret reveal {index}" if positive else f"Beginner tutorial step by step guide {index}"),
            "final_label": positive,
            "local_thumbnail": str(path),
            "thumbnail": "",
        })
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    artifacts = tmp_path / "artifacts"
    old_threads = torch.get_num_threads()
    try:
        monkeypatch.setattr(sys, "argv", [
            "compare_models.py", "--csv", str(csv_path), "--image-root", str(image_root),
            "--epochs", "1", "--batch-size", "4", "--patience", "1", "--max-samples", "40",
            "--threads", "1", "--artifact-dir", str(artifacts),
        ])
        compare_models.main()
        payload = json.loads((artifacts / "model_comparison_metrics.json").read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert {item["key"] for item in payload["models"]} == {"bigru_text", "cnn_image", "mlp_features", "multimodal_fusion"}
        for item in payload["models"]:
            assert 0 <= item["test_metrics"]["accuracy"] <= 1
            assert "confusion_matrix" in item["test_metrics"]
        assert (artifacts / "model_comparison.csv").is_file()
        assert (artifacts / "multimodal_model.pt").is_file()
        predictor = TrainedPredictor.load(
            model_path=artifacts / "multimodal_model.pt",
            vocab_path=artifacts / "vocab.json",
            config_path=artifacts / "model_config.json",
        )
        assert 0 <= predictor.predict_probability("How to learn step by step", None) <= 1
    finally:
        torch.set_num_threads(old_threads)
