"""Local media transcription and goal-fit analysis for Veri-Click.

This module uses faster-whisper for speech-to-text. Summary, context matching, and
content-specificity values are lightweight heuristics and are deliberately labeled as such.
Uploaded media is processed from a temporary file and is not persisted by this module.
"""
from __future__ import annotations

import base64
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

ALLOWED_MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpeg", ".mpg", ".m4v",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".wma",
}
MAX_MEDIA_BYTES = 200 * 1024 * 1024
MAX_TRANSCRIPT_CHARS = 30000

class MediaDependencyError(RuntimeError):
    """Raised when the optional media packages are not installed."""


class MediaProcessingError(RuntimeError):
    """Raised when a media file cannot be analyzed usefully."""


_STOPWORDS = set("""
a an the and or but if then than so because as at by for from in into of on onto to up with about after before between during through over under again further once here there when where why how all any both each few more most other some such no nor not only own same too very can will just should now i me my we our you your he she it they them this that these those is are was were be been being do does did have has had having am what which who whom whose their his her its also video videos watch watching worth learn learning tutorial guide explain explained show shows showed step steps example examples""".split())
_STOPWORDS.update("""mujhe mujheko hai hain ho hua hui ke ki ka ko se mein mai me aur ya par is us ye yeh woh wo kya kaise kyu kyun liye wala wali wale ek bhi nahi nahin kr kar sakta sakti sakte chahiye bata btao mujhe""".split())
_STOPWORDS.update("""है हैं हो हुआ हुई के की का को से में मे और या पर इस उस यह ये वह वो क्या कैसे क्यों लिए वाला वाली वाले एक भी नहीं नहिं था थी थे करना चाहिए मुझे तुम आप हम आपका आपकी अपने""".split())

_FILLER_WORDS = set("""um uh hmm like basically literally actually you know okay ok right so well guys hello hey welcome subscribe like comment share channel today video videos thing things stuff kinda sorta""".split())
_STRUCTURE_CUES = set("""step steps first second third example examples demonstrate demonstration method methods how why because result results data research tested test compare compared versus pros cons conclusion summary key takeaways formula code practice exercise learn build create fix solution""".split())


def _tokens(text: str) -> list[str]:
    # Keep Unicode combining marks attached to their letters so Devanagari and
    # other Indic scripts are not broken into partial syllables by stdlib re. 
    tokens: list[str] = []
    current: list[str] = []
    for char in (text or "").lower():
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or (category[0] == "M" and current):
            current.append(char)
        else:
            if current:
                token = "".join(current)
                if len(token) > 1:
                    tokens.append(token)
                current = []
    if current:
        token = "".join(current)
        if len(token) > 1:
            tokens.append(token)
    return tokens


def _content_tokens(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?।])\s+|\n+", normalized)
    sentences = [part.strip(" \t\r\n-•") for part in parts if len(part.strip()) >= 18]
    if not sentences and normalized:
        sentences = [normalized]
    return sentences


def extractive_summary(text: str, max_sentences: int = 4) -> list[str]:
    """Select transcript sentences using term salience; this is not abstractive generation."""
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences
    term_counts = Counter(_content_tokens(text))
    if not term_counts:
        return sentences[:max_sentences]
    max_count = max(term_counts.values())
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        words = _content_tokens(sentence)
        if not words:
            score = 0.0
        else:
            score = sum(term_counts[word] / max_count for word in words) / (len(words) ** 0.5)
        # Slightly favor the beginning because introductions often state the topic.
        score += max(0.0, 0.15 - index * 0.01)
        scored.append((score, index, sentence))
    picked = sorted(scored, reverse=True)[:max_sentences]
    return [sentence for _, _, sentence in sorted(picked, key=lambda item: item[1])]


# Goal matching is intentionally lexical (not a semantic embedding model). These
# small normalizations prevent obvious false misses such as "learning" vs "learn"
# and "basics" vs "basic". Intent/context words are omitted so "for my exam"
# does not unfairly penalize a useful lesson whose transcript never says "exam".
_CONTEXT_INTENT_STOPWORDS = set("""
want wants wanted need needs needed my our your their me us you we i exam exams
assessment assessments test tests revision revise revise preparation prepare
preparing studying study studying improve goal goal goals worth watching
""".split())
_RELEVANCE_STOPWORD_EXCEPTIONS = {"learn", "learning"}
_RELEVANCE_NORMALIZATION = {
    "learning": "learn", "learns": "learn", "learned": "learn", "learnt": "learn",
    "basics": "basic", "fundamentals": "fundamental", "networks": "network",
    "layers": "layer", "examples": "example", "tutorials": "tutorial",
    "explains": "explain", "explained": "explain", "explaining": "explain",
    "teaches": "teach", "teaching": "teach", "concepts": "concept",
}


def _relevance_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _tokens(text):
        if token in _CONTEXT_INTENT_STOPWORDS:
            continue
        if token in _STOPWORDS and token not in _RELEVANCE_STOPWORD_EXCEPTIONS:
            continue
        tokens.append(_RELEVANCE_NORMALIZATION.get(token, token))
    return tokens


def estimate_context_match(transcript: str, goal: str) -> dict[str, Any]:
    goal_terms = list(dict.fromkeys(_relevance_tokens(goal)))
    if not goal_terms:
        return {"score": None, "matched_terms": [], "goal_terms": [], "explanation": "No viewing goal was provided, so personal relevance cannot be estimated."}
    transcript_terms = set(_relevance_tokens(transcript))
    matched = [term for term in goal_terms if term in transcript_terms]
    score = round(100 * len(matched) / len(goal_terms))
    if score >= 60:
        explanation = "Several key terms from your goal appear in the transcript. Check the summary and key moments to confirm the context fits."
    elif score >= 30:
        explanation = "Some goal-related terms appear in the transcript, but the keyword match is partial. Preview the key moments before committing your time."
    else:
        explanation = "Few goal-related terms appear in the transcript. This is a lexical keyword estimate, not a semantic judgement; the video may still be useful."
    return {"score": score, "matched_terms": matched, "goal_terms": goal_terms, "explanation": explanation}


def estimate_content_specificity(transcript: str) -> dict[str, Any]:
    words = _tokens(transcript)
    content_words = [word for word in words if word not in _STOPWORDS]
    if not words:
        return {"score": 0, "label": "Not enough speech", "basis": "No transcript words were available."}
    diversity = len(set(content_words)) / max(1, len(content_words))
    filler_ratio = sum(1 for word in words if word in _FILLER_WORDS) / len(words)
    structure_hits = sum(1 for word in content_words if word in _STRUCTURE_CUES)
    diversity_score = min(100.0, diversity * 155)
    structure_score = min(100.0, structure_hits * 8)
    score = round(max(0, min(100, diversity_score * 0.65 + structure_score * 0.35 - min(30, filler_ratio * 400))))
    label = "More specific language signals" if score >= 60 else "Mixed specificity signals" if score >= 35 else "Limited specificity signals"
    return {
        "score": score,
        "label": label,
        "basis": "A heuristic based on vocabulary variety, instructional/detail cues, and common filler words—not a measure of factual accuracy or teaching quality.",
    }


def _safe_frame_data(frame: Any, timestamp: float) -> dict[str, Any]:
    image = frame.to_image().convert("RGB")
    image.thumbnail((480, 270))
    stats = ImageStat.Stat(image)
    brightness = sum(stats.mean[:3]) / (3 * 255)
    grayscale = image.convert("L")
    contrast = min(1.0, ImageStat.Stat(grayscale).stddev[0] / 100)
    saturation = ImageStat.Stat(image.convert("HSV")).mean[1] / 255
    output = BytesIO()
    image.save(output, format="JPEG", quality=68, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return {
        "timestamp_seconds": round(float(timestamp), 1),
        "timestamp_label": _timestamp_label(timestamp),
        "data_url": f"data:image/jpeg;base64,{encoded}",
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "saturation": round(saturation, 3),
    }


def _timestamp_label(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _media_metadata_and_frames(path: Path) -> tuple[float | None, bool, bool, list[dict[str, Any]]]:
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise MediaDependencyError("Media decoding dependency is missing. Run: pip install -r requirements-media.txt") from exc

    try:
        frames: list[dict[str, Any]] = []
        with av.open(str(path)) as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            if not audio_streams:
                raise MediaProcessingError("No audio track was found. Veri-Click needs spoken audio to evaluate what the video says.")
            duration = None
            if container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
            if duration is None or duration <= 0:
                durations = [float(s.duration * s.time_base) for s in list(audio_streams) + list(video_streams) if s.duration is not None and s.time_base is not None]
                duration = max(durations) if durations else None

            if video_streams:
                stream = video_streams[0]
                if duration and stream.time_base:
                    for fraction in (0.2, 0.5, 0.8):
                        target = max(0.0, min(duration - 0.05, duration * fraction))
                        try:
                            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
                            for frame in container.decode(video=0):
                                frame_time = float(frame.time) if frame.time is not None else target
                                if frame_time + 0.15 >= target:
                                    frames.append(_safe_frame_data(frame, frame_time))
                                    break
                        except Exception:
                            # A damaged video or unsupported seek should not prevent audio analysis.
                            continue
                elif not duration:
                    try:
                        container.seek(0, stream=stream, backward=True)
                        first = next(container.decode(video=0), None)
                        if first is not None:
                            frames.append(_safe_frame_data(first, float(first.time or 0.0)))
                    except Exception:
                        pass
        return duration, bool(video_streams), bool(audio_streams), frames
    except MediaProcessingError:
        raise
    except Exception as exc:
        raise MediaProcessingError("The file could not be decoded. Try an MP4, WebM, MP3, WAV, or M4A file that plays normally.") from exc


@lru_cache(maxsize=2)
def _load_whisper_model(model_size: str):
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise MediaDependencyError("Speech AI is not installed yet. In your activated .venv run: pip install -r requirements-media.txt") from exc
    # CPU int8 is chosen to reduce RAM use on student laptops.
    return WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=max(1, min(4, os.cpu_count() or 2)), num_workers=1)


def _key_moments(segments: list[dict[str, Any]], duration: float | None) -> list[dict[str, Any]]:
    if not segments:
        return []
    if not duration or duration < 60:
        windows = [(0.0, duration or max(seg["end"] for seg in segments), segments)]
    else:
        count = min(5, max(2, int(duration // 150) + 1))
        width = duration / count
        windows = []
        for index in range(count):
            start = index * width
            end = (index + 1) * width if index < count - 1 else duration + 0.1
            group = [seg for seg in segments if start <= seg["start"] < end]
            if group:
                windows.append((start, end, group))
    moments = []
    for start, end, group in windows:
        text = " ".join(seg["text"] for seg in group)
        summary = extractive_summary(text, max_sentences=1)
        excerpt = (summary[0] if summary else text).strip()
        if len(excerpt) > 260:
            excerpt = excerpt[:257].rstrip() + "..."
        moments.append({"timestamp_seconds": round(start, 1), "timestamp_label": _timestamp_label(start), "end_seconds": round(end, 1), "excerpt": excerpt})
    return moments[:5]


def analyze_media_file(path: Path, filename: str, title: str = "", goal: str = "") -> dict[str, Any]:
    """Transcribe an uploaded media file and estimate fit for a user-stated viewing goal."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_MEDIA_SUFFIXES:
        raise MediaProcessingError("Unsupported file type. Use MP4, MOV, MKV, WebM, AVI, MP3, WAV, M4A, AAC, OGG, FLAC, or WMA.")
    if not path.is_file() or path.stat().st_size == 0:
        raise MediaProcessingError("The uploaded file is empty or unavailable.")

    duration, has_video, has_audio, frames = _media_metadata_and_frames(path)
    if not has_audio:
        raise MediaProcessingError("No audio track was found in this file.")

    model_size = os.getenv("VERICLICK_WHISPER_MODEL", "tiny").strip() or "tiny"
    try:
        model = _load_whisper_model(model_size)
        whisper_segments, info = model.transcribe(
            str(path), beam_size=1, vad_filter=True, condition_on_previous_text=False,
        )
        segments: list[dict[str, Any]] = []
        transcript_parts: list[str] = []
        char_count = 0
        total_words = 0
        for segment in whisper_segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            segments.append({"start": float(segment.start), "end": float(segment.end), "text": text})
            transcript_parts.append(text)
            char_count += len(text) + 1
            total_words += len(_tokens(text))
            # Keep resource use and the response size bounded for very long uploads.
            if char_count >= MAX_TRANSCRIPT_CHARS:
                break
    except MediaDependencyError:
        raise
    except Exception as exc:
        raise MediaProcessingError("Speech transcription could not finish. Check your internet connection for the first tiny-model download, and try a shorter media file.") from exc

    transcript = " ".join(transcript_parts).strip()
    if len(_tokens(transcript)) < 8:
        raise MediaProcessingError("There was too little clear speech to assess. Try a clip with audible spoken content instead of only music or silence.")

    summary = extractive_summary(transcript, max_sentences=4)
    context = estimate_context_match(transcript, goal)
    specificity = estimate_content_specificity(transcript)
    title_promise_analysis = None
    if title.strip():
        try:
            from .title_promise import analyze_title_promise
            title_promise_analysis = analyze_title_promise(title, segments)
        except Exception as exc:
            # The core transcript report must remain usable if optional promise analysis fails.
            title_promise_analysis = {"fulfillment_status": "Not Enough Evidence", "extracted_promise": "Promise analysis unavailable", "reasoning": "The optional title-promise analysis did not complete.", "evidence_timestamps": [], "method": "Unavailable"}
    word_count = total_words
    if context["score"] is None:
        recommendation = "Add a viewing goal for a personal recommendation"
        worth_score = None
        recommendation_reason = "The speech was transcribed, but whether it is worth your time depends on what you want to learn or get from it."
    else:
        worth_score = round(context["score"] * 0.75 + specificity["score"] * 0.25)
        if context["score"] >= 60 and specificity["score"] >= 35:
            recommendation = "Likely worth a look for your goal"
        elif context["score"] >= 35:
            recommendation = "Maybe — preview the summary first"
        else:
            recommendation = "Weak match for your stated goal"
        recommendation_reason = context["explanation"]

    return {
        "filename": Path(filename).name,
        "title": title.strip()[:300],
        "goal": goal.strip()[:500],
        "media_kind": "video + audio" if has_video else "audio only",
        "has_video": has_video,
        "has_audio": has_audio,
        "duration_seconds": round(duration, 1) if duration else None,
        "duration_label": _timestamp_label(duration) if duration else "Unknown duration",
        "language": getattr(info, "language", "unknown") or "unknown",
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
        "transcription_model": f"Whisper {model_size} · CPU int8",
        "word_count": word_count,
        "transcript": transcript[:MAX_TRANSCRIPT_CHARS],
        "transcript_truncated": char_count >= MAX_TRANSCRIPT_CHARS,
        "summary": summary,
        "key_moments": _key_moments(segments, duration),
        "context_match_score": context["score"],
        "context_match_terms": context["matched_terms"],
        "context_goal_terms": context["goal_terms"],
        "context_explanation": context["explanation"],
        "content_specificity_score": specificity["score"],
        "content_specificity_label": specificity["label"],
        "content_specificity_basis": specificity["basis"],
        "worth_score": worth_score,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "sampled_frames": frames,
        "title_promise_analysis": title_promise_analysis,
        "frame_sampling_note": "Up to three frames are sampled for preview and basic brightness/contrast/saturation statistics; the app does not semantically understand every visual frame.",
        "disclaimer": "This is a preliminary relevance aid, not a guarantee of quality, factual accuracy, or entertainment value. Context matching is keyword-based; transcript quality depends on the audio and speech-recognition model. Uploaded media is deleted after analysis and the transcript is not saved to local history.",
    }
