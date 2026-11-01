from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch", reason="Training utilities require optional PyTorch")

from scripts.train_model import normalize_columns, _metrics


def test_training_csv_columns_are_normalized_and_bad_rows_removed():
    frame = pd.DataFrame({
        "Title": ["Sensational title", "A useful tutorial", "", "Invalid label"],
        "Clickbait": [1, 0, 1, "unknown"],
        "Image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    })
    normalized = normalize_columns(frame)
    assert list(normalized.columns) == ["video_title", "label", "thumbnail"]
    assert normalized["label"].tolist() == [1, 0]


def test_metrics_include_confusion_and_classification_report():
    result = _metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.2])
    assert result["accuracy"] == 0.5
    assert "precision" in result and "recall" in result and "f1" in result
    assert len(result["confusion_matrix"]) == 2
    assert "clickbait" in result["classification_report"]


def test_training_csv_requires_thumbnail_column():
    frame = pd.DataFrame({"title": ["a title"], "label": [1]})
    with pytest.raises(ValueError, match="thumbnail"):
        normalize_columns(frame)
