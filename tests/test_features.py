from __future__ import annotations

from vericlick.features import heuristic_risk, image_statistics, lexical_signals, numeric_feature_vector


def test_lexical_signals_detect_sensational_and_informative_markers():
    sensational = lexical_signals("STOP! You won't believe this SHOCKING secret!!!")
    informative = lexical_signals("How to replace a light switch: step-by-step tutorial")

    assert "shocking" in sensational["trigger_phrases"]
    assert sensational["exclamation_count"] == 4
    assert "how to" in informative["informative_phrases"]
    assert informative["word_count"] >= 5


def test_numeric_features_are_fixed_length_finite_and_bounded(sample_png_bytes):
    values = numeric_feature_vector("STOP! This shocking secret is revealed NOW!", sample_png_bytes)
    assert len(values) == 16
    assert all(isinstance(value, float) for value in values)
    assert all(0.0 <= value <= 1.0 for value in values)


def test_image_statistics_return_expected_fields(sample_png_bytes):
    stats = image_statistics(sample_png_bytes)
    assert {"brightness", "saturation", "contrast", "aspect_ratio", "edge_density"}.issubset(stats)
    assert stats["aspect_ratio"] == round(320 / 180, 4)
    assert 0 <= stats["brightness"] <= 1


def test_heuristic_is_explainable_and_score_is_bounded(sample_png_bytes):
    score, reasons, signals, visual = heuristic_risk("STOP! You won't believe this shocking secret?!", sample_png_bytes)
    assert 4 <= score <= 96
    assert reasons
    assert signals["trigger_phrases"]
    assert visual


def test_informative_title_gets_lower_demo_score_than_sensational_title():
    low, *_ = heuristic_risk("How to replace an electrical switch safely: step-by-step tutorial")
    high, *_ = heuristic_risk("STOP!!! You won't believe this shocking secret revealed NOW!!!")
    assert high > low
