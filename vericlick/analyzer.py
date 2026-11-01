from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from .features import heuristic_risk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _trained_predictor():
    try:
        from .deep_model import TrainedPredictor, trained_artifacts_present
        if trained_artifacts_present():
            return TrainedPredictor.load()
    except Exception as exc:  # the UI remains usable if optional DL weights/dependencies are absent
        logger.warning("Could not load trained model; using demo scorer: %s", exc)
    return None


def model_status() -> dict[str, Any]:
    predictor = _trained_predictor()
    if predictor is not None:
        return {"mode": "trained_model", "model_loaded": True, "model_name": "BiGRU + CNN gated fusion", "message": "A locally trained multimodal checkpoint is active."}
    try:
        from .deep_model import trained_artifacts_present
        artifacts = trained_artifacts_present()
    except Exception:
        artifacts = False
    return {
        "mode": "demo_heuristic",
        "model_loaded": False,
        "model_name": "Explainable offline demo scorer",
        "message": "No complete trained checkpoint is installed. This mode is useful for UI demos, not a research benchmark.",
        "incomplete_artifacts": artifacts,
    }


def analyze(title: str, image_bytes: bytes | None = None) -> dict[str, Any]:
    title = title.strip()
    if not title:
        raise ValueError("Please enter a video title.")
    predictor = _trained_predictor()
    if predictor is not None:
        risk_score = round(predictor.predict_probability(title, image_bytes) * 100)
        _, reasons, signals, visual = heuristic_risk(title, image_bytes)
        mode = "trained_model"
        model_name = "BiGRU + CNN gated fusion"
        disclaimer = "Prediction from the locally trained model. The accompanying reasons are descriptive cues, not feature-attribution explanations of the neural network. Check held-out test metrics before making research claims."
    else:
        risk_score, reasons, signals, visual = heuristic_risk(title, image_bytes)
        mode = "demo_heuristic"
        model_name = "Explainable offline demo scorer"
        disclaimer = "Demo risk index from interpretable rules and thumbnail style statistics; not a calibrated probability or a factual truth verdict."
    if risk_score >= 62:
        label, band, color = "High clickbait risk", "HIGH", "rose"
    elif risk_score >= 35:
        label, band, color = "Mixed signals · review", "REVIEW", "amber"
    else:
        label, band, color = "Likely informative", "LOW", "mint"
    return {
        "risk_score": int(max(0, min(100, risk_score))),
        "label": label,
        "risk_band": band,
        "status_color": color,
        "reasons": reasons[:8],
        "signals": signals,
        "visual_metrics": visual,
        "image_available": bool(image_bytes),
        "model_mode": mode,
        "model_name": model_name,
        "disclaimer": disclaimer,
        "components": {
            "sensational_language": min(100, round(100 * min(len(signals["trigger_phrases"]) / 2.0, 1.0))),
            "curiosity_gap": min(100, round(100 * min(len(signals["curiosity_words"]) / 3.0, 1.0))),
            "urgency": min(100, round(100 * min(len(signals["urgency_words"]) / 2.0, 1.0))),
            "visual_intensity": min(100, round(100 * ((visual.get("saturation", 0.0) + visual.get("contrast", 0.0)) / 1.4))) if visual else None,
        },
    }
