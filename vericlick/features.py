from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat

TRIGGER_PHRASES = (
    "you won't believe", "you will not believe",
    "shocking", "unbelievable", "insane", "mind blowing", "mind-blowing",
    "secret", "exposed", "gone wrong", "what happened next", "the truth about",
    "finally revealed", "changes everything", "before it's too late", "before it is too late",
    "never seen before", "must watch", "omg", "jaw dropping", "jaw-dropping",
    "this is why", "doctors hate", "they don't want you to know", "they don't want you to see",
    "biggest mistake", "breaking news", "life changing", "life-changing", "shocking truth",
)
CURIOSITY_WORDS = {
    "why", "what", "secret", "truth", "revealed", "happened", "actually", "finally",
    "mystery", "hidden", "unknown", "never", "this", "these", "those", "reason", "reasons",
}
URGENCY_WORDS = {
    "now", "today", "urgent", "immediately", "before", "stop", "don't", "dont", "must",
    "warning", "alert", "hurry", "quick", "quickly", "last chance", "deadline",
}
INFORMATIVE_PHRASES = (
    "step by step", "how to", "explained", "tutorial", "lecture", "full guide", "beginner guide",
    "official announcement", "product review", "lesson", "course overview", "research talk",
    "manual", "walkthrough", "documentary",
)
_POSITIVE = {"great", "best", "amazing", "love", "wonderful", "excellent", "fantastic", "win", "success"}
_NEGATIVE = {"worst", "hate", "terrible", "awful", "shocking", "disaster", "fail", "scam", "danger", "fear"}


def _title_tokens(title: str) -> list[str]:
    return re.findall(r"[\w']+", title.lower(), flags=re.UNICODE)


def _safe_image(image_bytes: bytes | None) -> Image.Image | None:
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            return source.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def image_statistics(image_bytes: bytes | None) -> dict[str, float]:
    """Cheap, deterministic thumbnail statistics; these are style cues, not semantic understanding."""
    image = _safe_image(image_bytes)
    if image is None:
        return {}
    original_width, original_height = image.size
    image.thumbnail((96, 96))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    mean_rgb = rgb.mean(axis=(0, 1))
    saturation = float(hsv[:, :, 1].mean())
    brightness = float(hsv[:, :, 2].mean())
    contrast = float(gray.std())
    colorfulness = float(np.mean(np.std(rgb, axis=(0, 1))))
    edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    edge_density = float((edges > 0.16).mean())
    width, height = original_width, original_height
    return {
        "brightness": round(brightness, 4),
        "saturation": round(saturation, 4),
        "contrast": round(contrast, 4),
        "colorfulness": round(colorfulness, 4),
        "edge_density": round(edge_density, 4),
        "aspect_ratio": round(width / max(height, 1), 4),
        "red_mean": round(float(mean_rgb[0]), 4),
        "green_mean": round(float(mean_rgb[1]), 4),
        "blue_mean": round(float(mean_rgb[2]), 4),
    }


def lexical_signals(title: str) -> dict[str, Any]:
    clean = title.strip()
    lower = clean.lower()
    tokens = _title_tokens(clean)
    words = [token for token in tokens if any(ch.isalnum() for ch in token)]
    trigger_hits = [phrase for phrase in TRIGGER_PHRASES if phrase in lower]
    curiosity_hits = sorted(set(words).intersection(CURIOSITY_WORDS))
    urgency_hits = sorted(set(words).intersection(URGENCY_WORDS))
    uppercase_words = [word for word in re.findall(r"\b[A-Z]{2,}\b", clean) if len(word) > 1]
    letters = [char for char in clean if char.isalpha()]
    uppercase_ratio = sum(char.isupper() for char in letters) / max(len(letters), 1)
    all_caps_ratio = len(uppercase_words) / max(len(words), 1)
    positive_hits = sorted(set(words).intersection(_POSITIVE))
    negative_hits = sorted(set(words).intersection(_NEGATIVE))
    sentiment_tone = (len(positive_hits) - len(negative_hits)) / max(len(words), 1)
    informative_hits = [phrase for phrase in INFORMATIVE_PHRASES if phrase in lower]
    emoji_count = sum(unicodedata.category(ch) in {"So", "Sk"} for ch in clean)
    return {
        "word_count": len(words),
        "trigger_phrases": trigger_hits,
        "curiosity_words": curiosity_hits,
        "urgency_words": urgency_hits,
        "uppercase_words": uppercase_words,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "all_caps_word_ratio": round(all_caps_ratio, 4),
        "exclamation_count": clean.count("!"),
        "question_count": clean.count("?"),
        "number_present": bool(re.search(r"\d", clean)),
        "emoji_count": emoji_count,
        "positive_words": positive_hits,
        "negative_words": negative_hits,
        "sentiment_tone": round(sentiment_tone, 4),
        "informative_phrases": informative_hits,
        "repeated_punctuation": bool(re.search(r"[!?]{2,}", clean)),
        "title_length": len(clean),
    }


def numeric_feature_vector(title: str, image_bytes: bytes | None = None) -> list[float]:
    """Fixed 16-dimensional vector shared by trainer and inference.

    The features are explanatory side-signals; they do not replace learned representations.
    """
    signals = lexical_signals(title)
    image = image_statistics(image_bytes)
    words = max(int(signals["word_count"]), 1)
    triggers = len(signals["trigger_phrases"])
    curiosity = len(signals["curiosity_words"])
    urgency = len(signals["urgency_words"])
    return [
        min(triggers / 3.0, 1.0),
        min(curiosity / 4.0, 1.0),
        min(urgency / 3.0, 1.0),
        float(signals["uppercase_ratio"]),
        min(int(signals["exclamation_count"]) / 3.0, 1.0),
        float(int(signals["question_count"]) > 0),
        float(bool(signals["number_present"])),
        float(signals["all_caps_word_ratio"]),
        min(words / 30.0, 1.0),
        min(int(signals["emoji_count"]) / 4.0, 1.0),
        float(len(signals["trigger_phrases"]) > 0),
        float(image.get("saturation", 0.0)),
        min(float(image.get("contrast", 0.0)) * 2.0, 1.0),
        float(image.get("brightness", 0.0)),
        min(float(image.get("colorfulness", 0.0)) * 2.0, 1.0),
        float(image.get("edge_density", 0.0)),
    ]


def heuristic_risk(title: str, image_bytes: bytes | None = None) -> tuple[int, list[str], dict[str, Any], dict[str, float]]:
    """Offline demo scorer. Its score is a transparent risk index, not a calibrated probability."""
    signals = lexical_signals(title)
    visual = image_statistics(image_bytes)
    score = 18.0
    score += min(len(signals["trigger_phrases"]), 3) * 13.0
    score += min(len(signals["curiosity_words"]), 3) * 4.5
    score += min(len(signals["urgency_words"]), 2) * 6.5
    score += min(int(signals["exclamation_count"]), 3) * 4.0
    score += 4.0 if signals["question_count"] else 0.0
    score += 8.0 if signals["uppercase_ratio"] > 0.42 and len(title) >= 8 else 0.0
    score += 4.0 if signals["all_caps_word_ratio"] > 0.18 else 0.0
    score += 3.0 if signals["number_present"] and len(signals["trigger_phrases"]) else 0.0
    score += 4.0 if signals["repeated_punctuation"] else 0.0
    score += 3.0 if signals["emoji_count"] >= 2 else 0.0
    score -= min(len(signals["informative_phrases"]), 2) * 8.0

    visual_flags: list[str] = []
    if visual:
        # These cues are weak and deliberately low-weight. Saturation is not evidence of deception by itself.
        if visual.get("saturation", 0.0) > 0.72:
            score += 3.0
            visual_flags.append("Highly saturated thumbnail palette")
        if visual.get("contrast", 0.0) > 0.27:
            score += 2.0
            visual_flags.append("High-contrast thumbnail composition")
    reasons: list[str] = []
    if signals["trigger_phrases"]:
        reasons.append("Sensational phrase(s): " + ", ".join(signals["trigger_phrases"][:4]))
    if signals["curiosity_words"]:
        reasons.append("Curiosity-gap wording: " + ", ".join(signals["curiosity_words"][:5]))
    if signals["urgency_words"]:
        reasons.append("Urgency language: " + ", ".join(signals["urgency_words"][:4]))
    if signals["uppercase_ratio"] > 0.42 and len(title) >= 8:
        reasons.append("Unusually high uppercase ratio")
    if signals["exclamation_count"] >= 2 or signals["repeated_punctuation"]:
        reasons.append("Repeated emphatic punctuation")
    if signals["number_present"]:
        reasons.append("Title contains a number or list-style cue")
    if signals["informative_phrases"]:
        reasons.append("Informative framing: " + ", ".join(signals["informative_phrases"][:3]))
    reasons.extend(visual_flags)
    if not reasons:
        reasons.append("No strong lexical clickbait markers detected")
    score = max(4, min(96, round(score)))
    return score, reasons, signals, visual
