"""Title Promise Checker and public YouTube-caption utilities.

The checker produces a transparent evidence-based estimate. Optional sentence-transformer
embeddings improve the ranking of transcript windows, but they are not an entailment model
and never establish factual truth. Lexical/evidence rules remain available without extras.
"""
from __future__ import annotations

import os
import re
import logging
from collections import Counter
from functools import lru_cache
from urllib.parse import parse_qs, urlencode, urlparse
from typing import Any

import requests

logger = logging.getLogger(__name__)
MAX_TRANSCRIPT_CHARS = 40000
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ALLOWED_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtube-nocookie.com", "youtube-nocookie.com"}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = set("""
a an the and or but if then than so because as at by for from in into of on onto to up with about after before between during through over under again further once here there when where why how all any both each few more most other some such no nor only own same too very can will just should now i me my we our you your he she it they them this that these those is are was were be been being do does did have has had having am what which who whom whose their his her its also video videos watch watching worth today now new latest actual really very your you re wont cant dont didnt does whats thats theyre its gonna gonna please``""".split())
_TITLE_GENERIC = set("""
how to what exactly is are was were why does did do can could would should i we you your my a an the of for in on with from by at this that these those all about best top ultimate complete full guide tutorial explained explanation beginner beginners basics fundamental fundamentals step steps ways tips tricks secret secrets shocking unbelievable unbelievable believe won t wont will never next happened crazy insane viral must watch reveal revealed exposing exposed honest reaction review reviewed test tested try tried testing versus vs comparison results result way tip trick secret today 2025 2026 4k official``""".split())
_PROMO_WORDS = set("""subscribe subscribed subscription like comment share bell channel sponsor sponsored sponsorship ad ads promotion promo code discount follow patreon merch members giveaway welcome intro outro smash notification notifications``""".split())
_FILLER_PATTERNS = (
    re.compile(r"\b(subscribe|smash that like|hit the bell|like and subscribe|follow me|check the link|use my code|sponsored by|thanks to our sponsor|welcome back to my channel)\b", re.I),
)
_STRUCTURAL_CUES = set("""first second third next then finally step steps example examples demonstrate demonstrates demonstration because therefore means defined definition explain explains explained explanation method methods process show shows showed test tested review compared compare conclusion summary result results why how formula proof evidence practice pros cons takeaways""".split())
_SYNONYM_GROUPS = {
    "learn": {"learn", "learning", "learned", "learnt", "teach", "teaches", "teaching", "taught", "lesson", "lessons", "course", "courses", "tutorial", "tutorials", "understand", "understanding", "explained", "explain"},
    "explain": {"explain", "explains", "explained", "explanation", "define", "defined", "definition", "means", "meaning", "overview", "describe", "describes", "described"},
    "step": {"step", "steps", "first", "second", "third", "then", "finally", "process", "method", "methods", "workflow", "procedure", "guide", "tutorial"},
    "review": {"review", "reviewed", "rating", "ratings", "pros", "cons", "tested", "testing", "verdict", "opinion", "worth", "recommend", "recommendation"},
    "compare": {"compare", "compared", "comparison", "versus", "vs", "difference", "differences", "better", "worse", "pros", "cons"},
    "make": {"make", "build", "create", "creating", "built", "made", "develop", "developed", "setup", "install", "configure"},
}
_NORMALIZE = {
    "learning": "learn", "learns": "learn", "learned": "learn", "learnt": "learn",
    "teaches": "teach", "teaching": "teach", "taught": "teach",
    "basics": "basic", "fundamentals": "fundamental", "networks": "network",
    "layers": "layer", "examples": "example", "tutorials": "tutorial",
    "explains": "explain", "explained": "explain", "explaining": "explain",
    "means": "mean", "defined": "define", "definition": "define",
    "demonstrates": "demonstrate", "demonstrated": "demonstrate", "demonstrating": "demonstrate",
    "reviews": "review", "reviewed": "review", "tested": "test", "testing": "test",
    "ways": "way", "tips": "tip", "tricks": "trick", "steps": "step",
    "methods": "method", "differences": "difference", "results": "result",
}


def extract_youtube_video_id(raw: str) -> str:
    """Accept a YouTube video URL or the 11-character video ID; reject other hosts."""
    raw = (raw or "").strip()
    if _VIDEO_ID_RE.fullmatch(raw):
        return raw
    try:
        parsed = urlparse(raw if "://" in raw else "https://" + raw)
    except ValueError as exc:
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in _ALLOWED_YT_HOSTS:
        raise ValueError("Only YouTube video links are supported (youtube.com or youtu.be).")
    video_id = ""
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        query = parse_qs(parsed.query)
        video_id = (query.get("v") or [""])[0]
        if not video_id:
            parts = [part for part in parsed.path.split("/") if part]
            for marker in ("embed", "shorts", "live", "v"):
                if marker in parts:
                    marker_index = parts.index(marker)
                    if marker_index + 1 < len(parts):
                        video_id = parts[marker_index + 1]
                        break
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError("Could not find a valid YouTube video ID in that link.")
    return video_id


def youtube_urls(video_id: str) -> dict[str, str]:
    if not _VIDEO_ID_RE.fullmatch(video_id or ""):
        raise ValueError("Invalid YouTube video ID.")
    return {
        "watch_url": f"https://www.youtube.com/watch?v={video_id}",
        "embed_url": f"https://www.youtube-nocookie.com/embed/{video_id}",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    }


def fetch_youtube_metadata(video_id: str) -> dict[str, str]:
    """Fetch basic public title/author metadata through YouTube's oEmbed endpoint."""
    url = youtube_urls(video_id)["watch_url"]
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=(3.0, 6.0),
            allow_redirects=False,
            headers={"User-Agent": "Veri-Click-AI/1.0"},
        )
        if response.status_code != 200:
            return {}
        payload = response.json()
        return {
            "title": str(payload.get("title") or "")[:500],
            "author_name": str(payload.get("author_name") or "")[:200],
            "thumbnail_url": str(payload.get("thumbnail_url") or youtube_urls(video_id)["thumbnail_url"]),
        }
    except (requests.RequestException, ValueError, TypeError) as exc:
        logger.info("YouTube metadata unavailable for %s: %s", video_id, exc)
        return {}


def fetch_youtube_transcript(video_id: str, languages: tuple[str, ...] = ("en", "hi", "en-US", "hi-IN")) -> list[dict[str, Any]]:
    """Fetch public captions only; does not download the video/audio itself."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YouTube captions support is not installed. Run: pip install -r requirements-promise.txt, or paste a timestamped transcript below.") from exc
    api = YouTubeTranscriptApi()
    transcript_obj = None
    errors: list[Exception] = []
    try:
        transcript_obj = api.fetch(video_id, languages=list(languages))
    except Exception as exc:
        errors.append(exc)
    if transcript_obj is None:
        # One default-language attempt is useful for versions with different language handling.
        try:
            transcript_obj = api.fetch(video_id)
        except Exception as exc:
            errors.append(exc)
    if transcript_obj is None:
        last = errors[-1] if errors else None
        raise RuntimeError(
            "Public captions could not be retrieved for this video. It may have captions disabled, age/region restrictions, or YouTube may be rate-limiting requests. Paste a timestamped transcript into the box and try again."
        ) from last
    snippets = getattr(transcript_obj, "snippets", transcript_obj)
    segments: list[dict[str, Any]] = []
    for item in snippets:
        text = str(getattr(item, "text", item.get("text", "") if isinstance(item, dict) else "") or "").replace("\n", " ").strip()
        if not text:
            continue
        start = float(getattr(item, "start", item.get("start", 0.0) if isinstance(item, dict) else 0.0) or 0.0)
        duration = float(getattr(item, "duration", item.get("duration", 0.0) if isinstance(item, dict) else 0.0) or 0.0)
        segments.append({"start": start, "end": start + max(0.0, duration), "text": text})
        if sum(len(s["text"]) for s in segments) >= MAX_TRANSCRIPT_CHARS:
            break
    if not segments:
        raise RuntimeError("The caption track was empty. Paste a transcript manually to continue.")
    return segments


def parse_timestamped_transcript(text: str) -> list[dict[str, Any]]:
    """Parse common timestamped transcript formats; plain text stays explicitly untimed."""
    text = (text or "").strip()[:MAX_TRANSCRIPT_CHARS]
    if not text:
        return []
    # Supports [01:20 - 01:45] text, 01:20 text, and 00:01:20.500 text.
    ts = r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)"
    pattern = re.compile(rf"^\s*\[?{ts}(?:\s*(?:-|–|to)\s*(?P<end>\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:\.\d{{1,3}})?))?\]?\s*(?:\||:|-)?\s*(?P<text>.+?)\s*$", re.I)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    segments: list[dict[str, Any]] = []
    all_timed = True
    for line in lines:
        match = pattern.match(line)
        if not match:
            all_timed = False
            break
        start = _parse_timestamp(match.group("time"))
        end = _parse_timestamp(match.group("end")) if match.group("end") else None
        snippet = match.group("text").strip()
        if not snippet:
            continue
        segments.append({"start": start, "end": end if end is not None else start + 4.0, "text": snippet})
    if all_timed and segments:
        return segments
    # An untimed transcript is preserved as evidence but does not get invented timestamps.
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [{"start": None, "end": None, "text": chunk.strip()} for chunk in chunks if chunk.strip()]


def _parse_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    parts = value.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        return 0.0


def _timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "Timestamp unavailable"
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _tokens(text: str) -> list[str]:
    return [_NORMALIZE.get(token.lower(), token.lower()) for token in _TOKEN_RE.findall(text or "") if len(token) > 1]


def _promise_terms(title: str) -> list[str]:
    tokens = _tokens(title)
    return list(dict.fromkeys(token for token in tokens if token not in _TITLE_GENERIC and token not in _STOP and token not in _PROMO_WORDS and not token.isdigit()))


def _display_topic(title: str, terms: list[str]) -> str:
    topic = (title or "").strip(" !?.:—-")
    prefix_patterns = (
        r"^(?:what exactly is|what is|what are|how does|how do|how to|why is|why are|why does|why do|can you|is it true that)\s+",
        r"^(?:top\s+\d+\s+(?:ways|tips|reasons|steps)\s+to|\d+\s+(?:ways|tips|reasons|steps)\s+to)\s+",
    )
    for pattern in prefix_patterns:
        topic = re.sub(pattern, "", topic, flags=re.I)
    topic = re.sub(r"^(?:learn|learning|explain|explaining|teach|teaching|understand|understanding|discover|find out)\s+", "", topic, flags=re.I)
    topic = re.sub(r"^(?:honest\s+)?review\s+of\s+", "", topic, flags=re.I)
    topic = re.sub(r"\s+(?:tutorial|guide|course)(?:\s+for\s+beginners?)?$", "", topic, flags=re.I)
    topic = re.sub(r"\s+for\s+beginners?$", "", topic, flags=re.I)
    topic = re.sub(r"^(?:the|a|an)\s+", "", topic, flags=re.I)
    topic = re.sub(r"\b(?:you won't believe|you will not believe|shocking|unbelievable|insane|viral|must watch|secret|revealed?)\b", "", topic, flags=re.I)
    topic = re.sub(r"[!?.,:;]+$", "", re.sub(r"\s+", " ", topic)).strip()
    return topic or " ".join(terms[:8]).strip() or (title or "").strip(" !?.:—-")


def extract_title_promise(title: str) -> dict[str, Any]:
    clean = re.sub(r"[\u200b\ufeff]", "", (title or "")).strip()
    lower = clean.lower()
    terms = _promise_terms(clean)
    topic = _display_topic(clean, terms)
    vague_clickbait = any(phrase in lower for phrase in ("you won't believe", "you will not believe", "what happened next", "shocking secret", "must see"))
    if vague_clickbait:
        promise = "The title teases a surprising event but does not state a specific, checkable content promise."
        kind = "vague_teaser"
    elif re.search(r"\bwhat\s+(?:exactly\s+)?(?:is|are)\b", lower):
        promise = f"Define or explain {topic or clean} with a clear answer and supporting detail."
        kind = "definition"
    elif re.search(r"\bwhy\b", lower):
        promise = f"Explain the reason(s) for {topic or clean}, not just mention the topic."
        kind = "why_explanation"
    elif re.search(r"\bhow\s+to\b|\bstep[- ]by[- ]step\b|\btutorial\b|\bguide\b", lower):
        promise = f"Provide actionable steps or a practical explanation for {topic or clean}."
        kind = "how_to"
    elif re.search(r"\b(review|worth it|pros and cons|vs\.?|versus|comparison)\b", lower):
        promise = f"Evaluate or compare {topic or clean} with reasons, trade-offs, or a conclusion."
        kind = "review_comparison"
    elif re.search(r"\b(top\s+\d+|\d+\s+(?:ways|tips|reasons|steps|mistakes))\b", lower):
        promise = f"Deliver the promised list of items about {topic or clean}."
        kind = "list"
    elif re.search(r"\b(learn|learning|explained|explain|teach|teaching|beginner|basics)\b", lower):
        promise = f"Teach or clearly explain {topic or clean}, with substantive content rather than a title mention alone."
        kind = "educational"
    else:
        promise = f"Provide substantive coverage of {topic or clean}."
        kind = "general"
    return {"extracted_promise": promise, "kind": kind, "core_terms": terms, "is_specific": bool(terms) and not vague_clickbait}


def _windows(segments: list[dict[str, Any]], window_seconds: float = 30.0) -> list[dict[str, Any]]:
    timed = [s for s in segments if s.get("start") is not None]
    untimed = [s for s in segments if s.get("start") is None]
    output: list[dict[str, Any]] = []
    if timed:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for segment in timed:
            bucket = int(float(segment.get("start", 0)) // window_seconds)
            buckets.setdefault(bucket, []).append(segment)
        for bucket in sorted(buckets):
            items = buckets[bucket]
            output.append({
                "start": min(float(item.get("start", bucket * window_seconds)) for item in items),
                "end": max(float(item.get("end", float(item.get("start", 0)) + 1)) for item in items),
                "text": " ".join(str(item.get("text", "")) for item in items).strip(),
            })
    elif untimed:
        # Preserve chunk boundaries for ranking but do not invent timestamps.
        for index in range(0, len(untimed), 3):
            output.append({"start": None, "end": None, "text": " ".join(str(x.get("text", "")) for x in untimed[index:index + 3]).strip()})
    return output


def _term_group(term: str) -> set[str]:
    for values in _SYNONYM_GROUPS.values():
        if term in values:
            return values
    return {term}


def _window_evidence(window_text: str, core_terms: list[str]) -> tuple[float, list[str]]:
    tokens = set(_tokens(window_text))
    matches: list[str] = []
    for term in core_terms:
        group = _term_group(term)
        if term in tokens or group.intersection(tokens):
            matches.append(term)
    coverage = len(matches) / max(1, len(core_terms))
    return coverage, matches


@lru_cache(maxsize=1)
def _load_sentence_model():
    from sentence_transformers import SentenceTransformer  # type: ignore
    model_name = os.getenv("VERICLICK_PROMISE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name, device="cpu")


def _semantic_window_scores(promise: str, window_texts: list[str]) -> list[float] | None:
    if not window_texts:
        return []
    try:
        model = _load_sentence_model()
        vectors = model.encode([promise, *window_texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return [float(max(-1.0, min(1.0, vectors[0].dot(vector)))) for vector in vectors[1:]]
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.info("Optional semantic promise model unavailable; using lexical evidence only: %s", exc)
        return None
    except Exception as exc:  # model download/runtimes vary by environment
        logger.info("Semantic model fell back to lexical evidence: %s", exc)
        return None


def analyze_title_promise(title: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the specified three-way promise status and transcript evidence map."""
    title = (title or "").strip()[:500]
    transcript_text = " ".join(str(segment.get("text", "")) for segment in segments).strip()
    words = _tokens(transcript_text)
    promise = extract_title_promise(title)
    core_terms = promise["core_terms"]
    windows = _windows(segments)
    if not title:
        raise ValueError("A video title is required for Title Promise Checker.")

    if not windows or not words:
        return {
            "extracted_promise": promise["extracted_promise"], "fulfillment_status": "Not Enough Evidence",
            "reasoning": "No usable transcript text was provided, so the title's promise cannot be checked.",
            "evidence_timestamps": [], "core_terms": core_terms, "matched_terms": [], "promise_kind": promise["kind"],
            "method": "No transcript evidence", "filler_detected": [], "transcript_word_count": len(words),
            "topic_term_coverage_percent": 0, "evidence_strength_percent": 0, "alignment_score_percent": 0,
            "semantic_similarity_index_percent": None, "semantic_model_used": False,
            "semantic_model_name": None, "analysis_engine": "No transcript evidence", "llm_used": False,
        }

    if not promise["is_specific"]:
        return {
            "extracted_promise": promise["extracted_promise"], "fulfillment_status": "Not Enough Evidence",
            "reasoning": "The title creates curiosity but does not make a specific, checkable promise. Transcript evidence alone cannot determine what exact expectation should be fulfilled.",
            "evidence_timestamps": [], "core_terms": core_terms, "matched_terms": [], "promise_kind": promise["kind"],
            "method": "Transparent title-claim rules", "filler_detected": [], "transcript_word_count": len(words),
            "topic_term_coverage_percent": 0, "evidence_strength_percent": 0, "alignment_score_percent": 0,
            "semantic_similarity_index_percent": None, "semantic_model_used": False,
            "semantic_model_name": None, "analysis_engine": "Transparent title-claim rules", "llm_used": False,
        }

    matched_all: list[str] = []
    window_coverages: list[float] = []
    window_matches: list[list[str]] = []
    for window in windows:
        coverage, matches = _window_evidence(window["text"], core_terms)
        window_coverages.append(coverage)
        window_matches.append(matches)
        matched_all.extend(matches)
    matched_terms = list(dict.fromkeys(matched_all))
    whole_coverage = len(set(matched_terms)) / max(1, len(core_terms))

    semantic_scores = _semantic_window_scores(promise["extracted_promise"], [w["text"][:1800] for w in windows])
    if semantic_scores is None:
        method = "Lexical evidence coverage + transparent content/filler rules"
        ranks = [coverage for coverage in window_coverages]
        max_semantic = None
    else:
        method = "Local sentence-transformer embeddings + lexical evidence rules (similarity is not entailment)"
        ranks = [0.70 * coverage + 0.30 * max(0.0, (score - 0.10) / 0.75) for coverage, score in zip(window_coverages, semantic_scores)]
        max_semantic = max(semantic_scores) if semantic_scores else 0.0

    substantive = []
    filler_segments = []
    for window in windows:
        text = window["text"]
        token_set = set(_tokens(text))
        is_filler = any(pattern.search(text) for pattern in _FILLER_PATTERNS) or (len(token_set.intersection(_PROMO_WORDS)) >= 3 and len(_tokens(text)) < 35)
        if is_filler:
            filler_segments.append(window)
        elif len(_tokens(text)) >= 8:
            substantive.append(window)
    filler_ratio = len(filler_segments) / max(1, len(windows))
    structural_hits = len(set(_tokens(transcript_text)).intersection(_STRUCTURAL_CUES))
    # A readable, deterministic 0-100 alignment index. This is deliberately not
    # presented as a probability: it combines whole-transcript topic-term coverage,
    # the strongest evidence windows, and a small filler penalty.
    top_evidence_ranks = sorted(ranks, reverse=True)[: min(3, len(ranks))]
    evidence_strength = sum(top_evidence_ranks) / max(1, len(top_evidence_ranks))
    alignment_fraction = (0.65 * whole_coverage + 0.35 * evidence_strength) * (1.0 - 0.25 * filler_ratio)
    alignment_score_percent = int(round(max(0.0, min(1.0, alignment_fraction)) * 100))
    topic_term_coverage_percent = int(round(100 * whole_coverage))
    evidence_strength_percent = int(round(100 * max(0.0, min(1.0, evidence_strength))))
    semantic_similarity_index_percent = (
        int(round(100 * max(0.0, min(1.0, max_semantic)))) if max_semantic is not None else None
    )
    semantic_model_name = os.getenv("VERICLICK_PROMISE_MODEL", "sentence-transformers/all-MiniLM-L6-v2") if semantic_scores is not None else None
    analysis_engine = (
        "Local sentence-transformer embeddings + transparent rules (not a generative LLM)"
        if semantic_scores is not None else
        "Lexical topic matching + transparent content/filler rules (no LLM or semantic model)"
    )
    question_or_definition = promise["kind"] in {"definition", "why_explanation"}
    direct_answer_cue = bool(re.search(r"\b(is|are|means|refers to|defined as|because|the reason|this happens|in simple terms|we call)\b", transcript_text, re.I))
    matched_coverage = whole_coverage
    evidence_windows = [i for i, score in enumerate(ranks) if score >= 0.18 and not any(windows[i] is filler for filler in filler_segments)]
    evidence_windows = sorted(evidence_windows, key=lambda i: ranks[i], reverse=True)
    best_rank = max(ranks) if ranks else 0.0

    if matched_coverage < 0.15 and best_rank < 0.30 and (max_semantic is None or max_semantic < 0.30):
        status = "Not Enough Evidence"
        reasoning = "The transcript contains little identifiable evidence for the core topic or claim in the title. This suggests the promise was not demonstrated in the available transcript, but missing/poor captions can also cause this result."
    elif question_or_definition and matched_coverage >= 0.5 and direct_answer_cue and len(words) >= 12 and filler_ratio < 0.6:
        status = "Fulfilled"
        reasoning = "The transcript returns to the title's core topic and includes a direct-answer or definition cue. The evidence below shows where the promise is addressed; this is transcript matching, not a truth verification."
    elif matched_coverage >= 0.65 and len(words) >= 45 and (len(substantive) >= 2 or structural_hits >= 2) and filler_ratio < 0.45:
        status = "Fulfilled"
        reasoning = "Multiple title-topic terms appear with substantive detail or explanatory structure in the transcript. The listed timestamps are the strongest supporting windows."
    elif matched_coverage >= 0.18 or best_rank >= 0.30 or (max_semantic is not None and max_semantic >= 0.43):
        status = "Partly Fulfilled"
        if filler_ratio >= 0.4:
            reasoning = "The transcript touches on the promised topic, but substantial filler or promotional language reduces the evidence that the main promise is delivered in depth."
        else:
            reasoning = "The transcript contains some relevant evidence, but the available text does not clearly demonstrate the whole promise with enough depth or structure to mark it fulfilled."
    else:
        status = "Not Enough Evidence"
        reasoning = "The transcript does not provide enough clear evidence for the title's promise. Automated captions may be incomplete, so review the video before treating this as a misleading-title finding."

    if status == "Fulfilled" and filler_ratio >= 0.4:
        status = "Partly Fulfilled"
        reasoning = "Relevant content appears, but a high share of the sampled transcript windows contains filler or promotional language, so full fulfilment is not established."

    evidence: list[dict[str, str]] = []
    for idx in evidence_windows[:3]:
        window = windows[idx]
        raw = re.sub(r"\s+", " ", window["text"]).strip()
        if len(raw) > 300:
            raw = raw[:297].rsplit(" ", 1)[0] + "…"
        start = window.get("start")
        end = window.get("end")
        timestamp_range = f"[{_timestamp(start)} - {_timestamp(end)}]" if start is not None and end is not None else "Timestamp unavailable (untimed transcript)"
        evidence.append({"timestamp_range": timestamp_range, "summary": raw or "Relevant transcript window", "matched_terms": ", ".join(window_matches[idx])})
    filler_detected: list[str] = []
    for filler in filler_segments[:5]:
        snippet = re.sub(r"\s+", " ", filler["text"]).strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "…"
        filler_detected.append(snippet)

    return {
        "extracted_promise": promise["extracted_promise"],
        "fulfillment_status": status,
        "reasoning": reasoning,
        "evidence_timestamps": evidence,
        "core_terms": core_terms,
        "matched_terms": list(dict.fromkeys(matched_terms)),
        "promise_kind": promise["kind"],
        "method": method,
        "semantic_similarity_available": semantic_scores is not None,
        "filler_detected": filler_detected,
        "filler_window_ratio": round(filler_ratio, 3),
        "transcript_word_count": len(words),
        "transcript_coverage": round(matched_coverage, 3),
        "max_semantic_similarity": round(max_semantic, 3) if max_semantic is not None else None,
        "topic_term_coverage_percent": topic_term_coverage_percent,
        "evidence_strength_percent": evidence_strength_percent,
        "alignment_score_percent": alignment_score_percent,
        "alignment_score_definition": "0-100 composite index: 65% title-term coverage + 35% strongest-window evidence, with a 25%-scaled filler penalty; not a probability.",
        "semantic_similarity_index_percent": semantic_similarity_index_percent,
        "semantic_model_used": semantic_scores is not None,
        "semantic_model_name": semantic_model_name,
        "analysis_engine": analysis_engine,
        "llm_used": False,
    }



_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_LIST_UNITS = {
    "tool", "tools", "tip", "tips", "way", "ways", "step", "steps",
    "reason", "reasons", "mistake", "mistakes", "app", "apps", "website",
    "websites", "site", "sites", "method", "methods", "habit", "habits",
    "strategy", "strategies", "idea", "ideas", "lesson", "lessons", "rule",
    "rules", "fact", "facts", "feature", "features", "benefit", "benefits",
    "example", "examples", "secret", "secrets", "trick", "tricks", "plugin",
    "plugins", "extension", "extensions", "book", "books", "program", "programs",
}
_CLAIM_QUALIFIERS = {"free", "best", "easy", "simple", "fast", "quick", "cheap", "cheaply", "beginner", "beginners", "ultimate", "guaranteed"}
_CLAIM_METADATA_RE = re.compile(r"\b(?:chapter|part|episode|ep\.?|season)\s*#?\s*\d+\b", re.I)
_FREE_PATTERNS = (
    re.compile(r"\bfree\b", re.I),
    re.compile(r"\bno\s+(?:cost|charge|payment)\b", re.I),
    re.compile(r"\bwithout\s+pay(?:ing)?\b", re.I),
    re.compile(r"\bcosts?\s+nothing\b", re.I),
    re.compile(r"\bfree\s+of\s+charge\b", re.I),
)
_OUTCOME_RE = re.compile(
    r"\b(?P<action>save|saves|saving|saved|earn|earns|earning|make|makes|making|lose|loses|losing|gain|gains|gaining|reduce|reduces|reducing|cut|cuts|cutting|learn|learns|learning|build|builds|building|get|gets|getting|increase|increases|increasing)\s+"
    r"(?P<amount>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s*"
    r"(?P<unit>hours?|minutes?|seconds?|days?|weeks?|months?|years?|dollars?|rupees?|percent|%)\b",
    re.I,
)
_DURATION_RE = re.compile(r"\b(?:in|within)\s+(?P<amount>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s*(?P<unit>hours?|minutes?|seconds?|days?|weeks?)\b", re.I)


def _claim_terms(text: str, *, include_numbers: bool = False) -> list[str]:
    excluded = _STOP | {"chapter", "part", "episode", "season", "top", "free", "best", "easy", "simple", "quick", "fast", "ultimate", "guaranteed", "without", "using", "use", "used", "that", "which", "who", "with", "from", "for", "into", "your", "you", "video"}
    tokens = _tokens(text)
    return list(dict.fromkeys(
        token for token in tokens
        if token not in excluded and (include_numbers or not token.isdigit())
    ))


def _count_from_text(value: str) -> int | None:
    value = (value or "").lower().strip()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _topic_claim(text: str, claim_id: str) -> dict[str, Any] | None:
    clean = re.sub(r"\s+", " ", _CLAIM_METADATA_RE.sub("", (text or ""))).strip(" -–—|:;,.")
    clean = re.sub(r"^(?:what\s+(?:exactly\s+)?(?:is|are)|how\s+to|how\s+(?:does|do)|why\s+(?:does|do|is|are)|learn|learning|explained?|a\s+guide\s+to|guide\s+to)\s+", "", clean, flags=re.I)
    clean = re.sub(r"\b(?:that|which)\s+(?:saves?|save|earns?|earn|makes?|make|helps?|help|will|can)\b.*$", "", clean, flags=re.I).strip(" -–—|:;,.")
    terms = _claim_terms(clean)
    if not terms:
        return None
    if len(terms) == 1 and terms[0] in {"deep", "ai", "python", "java", "coding", "programming"}:
        pass
    if re.search(r"\bhow\b|\bwhy\b|\bwhat\b", text or "", re.I):
        claim_text = f'The video explains the topic “{clean}”.'
    else:
        claim_text = f'The video substantively covers “{clean}”.'
    return {
        "claim_id": claim_id, "claim_text": claim_text, "claim_type": "topic",
        "terms": terms, "topic_terms": terms, "required_count": None,
        "action": None, "amount": None, "unit": None,
    }


def extract_title_claims(title: str) -> list[dict[str, Any]]:
    """Split common, checkable title promises into auditable claims using transparent rules.

    This is deliberately not an LLM: it handles common list/count, price, quantified
    outcome, duration, and topic patterns and reports its decomposition for review.
    """
    clean_title = re.sub(r"[\u200b\ufeff]", "", (title or "")).strip()[:500]
    if not clean_title:
        return []
    if any(phrase in clean_title.lower() for phrase in ("you won't believe", "you will not believe", "what happened next", "shocking secret", "must see")):
        return [{
            "claim_id": "C1",
            "claim_text": "The title teases a surprising event but does not state a specific, checkable claim.",
            "claim_type": "vague_teaser", "terms": [], "topic_terms": [],
            "required_count": None, "action": None, "amount": None, "unit": None,
        }]

    claims: list[dict[str, Any]] = []

    def add(claim: dict[str, Any] | None) -> None:
        if not claim:
            return
        if len(claims) >= 6:
            return
        signature = (claim["claim_type"], tuple(claim.get("terms") or []), claim.get("required_count"), claim.get("action"), claim.get("amount"), claim.get("unit"))
        for existing in claims:
            existing_signature = (existing["claim_type"], tuple(existing.get("terms") or []), existing.get("required_count"), existing.get("action"), existing.get("amount"), existing.get("unit"))
            if signature == existing_signature:
                return
        claims.append(claim)

    # Common count/list promise, e.g. "5 free AI tools that save 10 hours".
    list_re = re.compile(
        r"\b(?P<count>\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)\s+"
        r"(?P<between>[^,|;:]{0,48}?)\b(?P<unit>tools?|tips?|ways?|steps?|reasons?|mistakes?|apps?|websites?|sites?|methods?|habits?|strategies|ideas?|lessons?|rules?|facts?|features?|benefits?|examples?|secrets?|tricks?|plugins?|extensions?|books?|programs?)\b",
        re.I,
    )
    list_match = list_re.search(clean_title)
    if list_match:
        count = _count_from_text(list_match.group("count"))
        unit = list_match.group("unit").lower()
        before_unit = list_match.group("between")
        subject_words = re.sub(r"\b(?:free|best|easy|simple|quick|fast|cheap|ultimate|guaranteed|top)\b", "", before_unit, flags=re.I)
        subject_words = re.sub(r"\s+", " ", subject_words).strip()
        tail = clean_title[list_match.end():]
        # For "10 ways to learn Python", the topic follows the list noun.
        topic_tail_match = re.match(r"\s+to\s+(.+?)(?=\s+(?:in|within)\s+\d+|$)", tail, flags=re.I)
        if topic_tail_match and unit.rstrip("s") in {"way", "step", "reason", "method", "strategy", "idea", "lesson", "rule", "fact", "example", "tip", "mistake"}:
            topic = topic_tail_match.group(1).strip(" .:;|-–—")
        else:
            # Trim a trailing result promise: "AI tools that save 10 hours" -> "AI tools".
            topic = re.sub(r"\b(?:that|which)\s+(?:save|saves|earn|earns|make|makes|help|helps|will|can|let|lets)\b.*$", "", subject_words + " " + unit, flags=re.I).strip()
            if not topic or not _claim_terms(topic):
                topic = unit
        add(_topic_claim(topic, "C1"))
        if count is not None:
            add({
                "claim_id": f"C{len(claims) + 1}",
                "claim_text": f"The video presents {count} {unit} as promised by the title.",
                "claim_type": "list_count", "terms": _claim_terms(unit),
                "topic_terms": _claim_terms(topic), "required_count": count,
                "action": None, "amount": None, "unit": unit,
            })

    if re.search(r"\bfree\b", clean_title, re.I):
        topic_terms = list(dict.fromkeys(token for token in (claims[0].get("topic_terms", []) if claims else _claim_terms(clean_title)) if token not in {"free", "ai"}))
        add({
            "claim_id": f"C{len(claims) + 1}",
            "claim_text": "The advertised item or method is presented as free or available at no cost.",
            "claim_type": "free", "terms": ["free", *topic_terms],
            "topic_terms": topic_terms, "required_count": None,
            "action": None, "amount": None, "unit": None,
        })

    outcome_match = _OUTCOME_RE.search(clean_title)
    if outcome_match:
        action = outcome_match.group("action").lower()
        amount = _count_from_text(outcome_match.group("amount"))
        unit = outcome_match.group("unit").lower()
        pre_action = clean_title[:outcome_match.start()].strip(" -–—|:;,.")
        topic_terms = _claim_terms(pre_action)
        add({
            "claim_id": f"C{len(claims) + 1}",
            "claim_text": f'The video provides evidence for the promised outcome “{action} {outcome_match.group("amount")} {unit}”.',
            "claim_type": "quantified_outcome", "terms": [action, outcome_match.group("amount").lower(), unit.rstrip("s")],
            "topic_terms": topic_terms, "required_count": None,
            "action": action, "amount": amount, "unit": unit,
        })

    duration_match = _DURATION_RE.search(clean_title)
    if duration_match:
        amount = _count_from_text(duration_match.group("amount"))
        unit = duration_match.group("unit").lower()
        topic_terms = _claim_terms(clean_title[:duration_match.start()].strip(" -–—|:;,."))
        add({
            "claim_id": f"C{len(claims) + 1}",
            "claim_text": f'The video supports the promised time frame “in {duration_match.group("amount")} {unit}”.',
            "claim_type": "duration", "terms": [duration_match.group("amount").lower(), unit.rstrip("s")],
            "topic_terms": topic_terms, "required_count": None,
            "action": None, "amount": amount, "unit": unit,
        })

    if re.search(r"\b(?:without\s+coding|no[- ]code|no coding|zero coding)\b", clean_title, re.I):
        add({
            "claim_id": f"C{len(claims) + 1}",
            "claim_text": "The promised task can be done without coding.",
            "claim_type": "no_code", "terms": ["coding", "without"],
            "topic_terms": _claim_terms(clean_title), "required_count": None,
            "action": None, "amount": None, "unit": None,
        })

    # A duration/outcome qualifier should not replace the core topic claim.
    if claims and not any(claim["claim_type"] == "topic" for claim in claims):
        core = clean_title
        if outcome_match:
            core = clean_title[:outcome_match.start()].strip(" -–—|:;,")
        elif duration_match:
            core = clean_title[:duration_match.start()].strip(" -–—|:;,")
        if list_match and not core:
            core = clean_title
        topic_claim = _topic_claim(core or clean_title, "C1")
        if topic_claim:
            claims.insert(0, topic_claim)

    # If no special shape was recognized, split on title separators and audit distinct topic clauses.
    if not claims:
        parts = [re.sub(r"\s+", " ", part).strip(" -–—|:;,." ) for part in re.split(r"\s*(?:\||—|–|:|;|,)\s*", clean_title)]
        parts = [part for part in parts if part and not _CLAIM_METADATA_RE.search(part)]
        for part in parts:
            add(_topic_claim(part, f"C{len(claims) + 1}"))
        if not claims:
            add(_topic_claim(clean_title, "C1"))

    # Use sequential IDs after any duplicate removal.
    for index, claim in enumerate(claims, start=1):
        claim["claim_id"] = f"C{index}"
    return claims[:6]


def _claim_window_score(claim: dict[str, Any], window: dict[str, Any]) -> tuple[float, list[str], str]:
    text = str(window.get("text", ""))
    tokens = set(_tokens(text))
    claim_type = claim["claim_type"]
    terms = list(claim.get("terms") or [])
    topic_terms = list(claim.get("topic_terms") or [])
    matched = [term for term in terms if term in tokens or _term_group(term).intersection(tokens)]
    topic_matched = [term for term in topic_terms if term in tokens or _term_group(term).intersection(tokens)]

    if claim_type == "vague_teaser":
        return 0.0, [], "This title is a vague teaser and does not state a specific claim to audit."

    if claim_type == "topic":
        coverage = len(matched) / max(1, len(terms))
        word_count = len(_tokens(text))
        substance_cues = set(_tokens(text)).intersection(_STRUCTURAL_CUES | {"introduce", "introducing", "introduction", "teach", "teaches", "teaching", "learn", "learning", "works", "work", "show", "shows", "showing", "use", "uses", "using", "example", "examples", "because", "means"})
        if coverage and word_count >= 8 and substance_cues:
            return coverage, matched, "Topic terms appear in a substantive transcript window with explanatory or structural cues."
        if coverage and word_count >= 8:
            return coverage * 0.65, matched, "Topic terms appear, but the window has limited explanatory cues; review context before treating this as support."
        return coverage * 0.35, matched, "Topic words appear only briefly, so this is a mention rather than clear substantive support." if matched else "No title-topic terms were found in this window."

    if claim_type == "list_count":
        count = int(claim.get("required_count") or 0)
        unit = str(claim.get("unit") or "")
        count_forms = {str(count), next((word for word, number in _NUMBER_WORDS.items() if number == count), "")}
        count_present = any(re.search(rf"\b{re.escape(form)}\b", text, re.I) for form in count_forms if form)
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        marker_count = sum(1 for marker in ordinals[:min(count, len(ordinals))] if re.search(rf"\b{marker}\b", text, re.I))
        unit_pattern = re.compile(rf"\b{re.escape(unit.rstrip('s'))}s?\b", re.I)
        unit_present = bool(unit_pattern.search(text))
        topic_present = bool(topic_matched) or unit_present
        if unit_present and count > 0 and marker_count >= min(count, len(ordinals)):
            return 1.0, list(dict.fromkeys([*matched, unit, *ordinals[:marker_count]])), f"The transcript contains list markers for all {count} promised items."
        if unit_present and count > 0 and marker_count:
            score = min(0.85, marker_count / max(1, count))
            return score, list(dict.fromkeys([*matched, unit, *ordinals[:marker_count]])), f"The transcript contains list markers for {marker_count} of {count} promised items."
        if unit_present and count_present and topic_present:
            return 0.55, list(dict.fromkeys([*matched, unit, str(count)])), "The title's count and item type are mentioned, but the transcript does not independently establish a complete list."
        if unit_present and count_present:
            return 0.45, list(dict.fromkeys([*matched, unit, str(count)])), "The count and item type are present, though a complete item-by-item list was not established."
        if unit_present or count_present:
            return 0.35, list(dict.fromkeys([*matched, unit if unit_present else str(count)])), "The transcript mentions part of the promised list/count, but not both together."
        return 0.0, matched, "No matching list count or item type was found."

    if claim_type == "free":
        qualifier = any(pattern.search(text) for pattern in _FREE_PATTERNS)
        price_caveat = bool(re.search(r"\b(?:some|only|limited|trial|however|but|while|whereas|paid|payment|requires?\s+(?:a\s+)?(?:paid|subscription)|premium\s+(?:features|plan)|not\s+all)\b", text, re.I))
        if qualifier and (topic_matched or not topic_terms) and price_caveat:
            return 0.55, list(dict.fromkeys([*topic_matched, "free / no-cost", "pricing caveat"])), "A free/no-cost option is mentioned, but the same window includes a pricing or availability caveat. Do not infer that every advertised item is fully free."
        if qualifier and (topic_matched or not topic_terms):
            return 1.0, list(dict.fromkeys([*topic_matched, "free / no-cost"])), "The transcript explicitly describes the item or method as free or no-cost."
        if qualifier:
            return 0.6, ["free / no-cost"], "The transcript mentions a free/no-cost option, but linkage to the title's topic is weak."
        if any(token in tokens for token in ("price", "pricing", "paid", "payment", "cost", "subscription")):
            return 0.25, list(dict.fromkeys(topic_matched)), "The transcript discusses pricing, but does not explicitly establish that the item is free."
        return 0.0, [], "No explicit free/no-cost evidence was found in this window."

    if claim_type == "quantified_outcome":
        action = str(claim.get("action") or "")
        amount = claim.get("amount")
        unit = str(claim.get("unit") or "")
        action_variants = {
            "save": ("save", "saves", "saving", "saved", "reduce", "reduces", "reduced", "cut", "cuts"),
            "earn": ("earn", "earns", "earned", "earning", "make", "makes", "made"),
            "make": ("make", "makes", "made", "earn", "earns", "earned"),
            "lose": ("lose", "loses", "lost", "losing", "reduce", "reduces"),
            "gain": ("gain", "gains", "gained", "increase", "increases"),
            "reduce": ("reduce", "reduces", "reduced", "cut", "cuts", "save", "saves"),
            "cut": ("cut", "cuts", "cutting", "reduce", "reduces"),
            "learn": ("learn", "learns", "learned", "learning", "teach", "teaches", "taught"),
            "build": ("build", "builds", "built", "building", "create", "creates"),
            "get": ("get", "gets", "got", "obtain", "receive"),
            "increase": ("increase", "increases", "increased", "boost", "boosts", "raise", "raises"),
        }.get(action, (action,))
        action_present = any(re.search(rf"\b{re.escape(value)}\b", text, re.I) for value in action_variants)
        number_forms = {str(amount), next((word for word, number in _NUMBER_WORDS.items() if number == amount), "")} if amount is not None else set()
        number_present = any(re.search(rf"\b{re.escape(value)}\b", text, re.I) for value in number_forms if value)
        unit_present = bool(re.search(rf"\b{re.escape(unit.rstrip('s'))}s?\b|%", text, re.I))
        if action_present and number_present and unit_present:
            denial = re.search(r"\b(?:not\s+(?:guaranteed|promised|possible|true|always)|no\s+guarantee|cannot|can't|won't|will\s+not|doesn't|does\s+not|never|not\s+necessarily)\b", text, re.I)
            # If the same window explicitly denies/qualifies the outcome, mention alone is not support.
            if denial:
                return 0.10, list(dict.fromkeys([action, str(amount), unit, "explicit caveat/denial"])), "The transcript mentions the quantity but also explicitly denies or disclaims the outcome; mention alone is not supporting evidence."
            if re.search(r"\b(?:may|might|could|can|depends?|varies?|in some cases|not guaranteed|not always)\b", text, re.I):
                return 0.55, list(dict.fromkeys([action, str(amount), unit, "qualified claim"])), "The outcome is mentioned with qualifying language, so the title's exact result is only partly supported."
            return 1.0, list(dict.fromkeys([action, str(amount), unit])), "The transcript mentions the action, quantity, and unit in the same window."
        if action_present and (number_present or unit_present) or (number_present and unit_present):
            return 0.5, list(dict.fromkeys([action if action_present else "", str(amount) if number_present else "", unit if unit_present else ""])), "Some parts of the quantified outcome appear, but the full amount/action combination is not established."
        return 0.0, [], "No matching action-plus-quantity evidence was found in this window."
    if claim_type == "duration":
        amount = claim.get("amount")
        unit = str(claim.get("unit") or "")

        number_forms = {
            str(amount),
            next(
                (word for word, number in _NUMBER_WORDS.items()
                 if number == amount),
                ""
            )
        } if amount is not None else set()

        number_present = any(
            re.search(rf"\b{re.escape(value)}\b", text, re.I)
            for value in number_forms if value
        )

        unit_present = bool(
            re.search(
                rf"\b{re.escape(unit.rstrip('s'))}s?\b",
                text,
                re.I
            )
        )

        topic_present = bool(topic_matched)

        completion_cue = bool(re.search(
            r"\b(?:you can now|now you can|you should now be able|"
            r"you have learned|you've learned|we have covered|"
            r"we've covered|to recap|in conclusion)\b",
            text,
            re.I
        ))

        matched = list(dict.fromkeys([
            *([str(amount)] if number_present else []),
            *([unit] if unit_present else []),
            *topic_matched
        ]))

        if number_present and unit_present and topic_present:
            if completion_cue:
                return (
                    1.0,
                    matched,
                    "The duration, topic, and a completion cue appear "
                    "in the transcript."
                )

            return (
                0.55,
                matched,
                "The transcript mentions the promised duration and topic, "
                "but this alone does not establish that the promised outcome "
                "was achieved within that time."
            )

        if number_present and unit_present:
            return (
                0.55,
                matched,
                "The duration is mentioned, but its connection to the "
                "promised topic or outcome is not fully established."
            )

        if unit_present or topic_present:
            return (
                0.25,
                matched,
                "Only partial evidence for the promised duration or "
                "topic was found."
            )

        return 0.0, [], "No matching duration evidence was found."

    if claim_type == "no_code":
        if re.search(
            r"\b(?:no[- ]code|no coding|without coding|"
            r"without writing code|does not require coding|"
            r"don't need to code)\b",
            text,
            re.I
        ):
            return (
                1.0,
                ["no-code / without coding"],
                "The transcript explicitly says coding is not required."
            )

        if re.search(r"\b(code|coding|programming)\b", text, re.I):
            return (
                0.25,
                ["coding"],
                "Coding is discussed, but the transcript does not establish "
                "the promised no-code condition."
            )

        return 0.0, [], "No evidence for the no-code condition was found."

    coverage = len(matched) / max(1, len(terms))

    return (
        coverage,
        matched,
        "Some claim terms appear in the transcript."
        if matched
        else "No matching claim terms were found."
    )

def analyze_title_claims(title: str, segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Run PromiseProof claim-wise transcript evidence audit; not factual verification."""
    claims = extract_title_claims(title)
    windows = _windows(segments)
    transcript_text = " ".join(str(segment.get("text", "")) for segment in segments)
    word_count = len(_tokens(transcript_text))
    if not claims:
        return {
            "claims": [], "total_claims": 0, "supported_count": 0,
            "partly_supported_count": 0, "no_evidence_count": 0,
            "evidence_coverage_percent": 0, "fully_supported_percent": 0,
            "limited_evidence": True,
            "limitation_note": "No specific checkable title claims could be extracted. Review the title manually.",
            "method": "Transparent title-claim rules + transcript matching (no generative LLM)",
            "llm_used": False, "semantic_model_used": False,
        }

    semantic_available: bool | None = None
    public_claims: list[dict[str, Any]] = []
    for claim in claims:
        lexical_scores: list[float] = []
        per_window: list[tuple[float, list[str], str]] = []
        for window in windows:
            per_window.append(_claim_window_score(claim, window))
            lexical_scores.append(per_window[-1][0])
        # The whole-transcript score complements the strongest timestamped evidence window.
        claim_type = claim["claim_type"]
        if lexical_scores:
            best_index = max(range(len(lexical_scores)), key=lambda idx: lexical_scores[idx])
            best_lexical = lexical_scores[best_index]
            if claim_type in {"topic"}:
                union_tokens = set(_tokens(transcript_text))
                terms = list(claim.get("terms") or [])
                total_coverage = sum(1 for term in terms if term in union_tokens or _term_group(term).intersection(union_tokens)) / max(1, len(terms))
                lexical_score = 0.55 * total_coverage + 0.45 * best_lexical
            else:
                lexical_score = best_lexical
        else:
            best_index, best_lexical, lexical_score = -1, 0.0, 0.0

        semantic_score = None
        # Optional semantic similarity can help rank topic evidence, but cannot overrule explicit
        # numeric/free/no-code checks: embedding similarity is not logical entailment.
        if windows and claim_type == "topic":
            sem = _semantic_window_scores(claim["claim_text"], [window["text"][:1800] for window in windows])
            if sem is not None:
                semantic_available = True
                semantic_score = max(sem) if sem else 0.0
                normalized_semantic = max(0.0, min(1.0, (semantic_score - 0.10) / 0.75))
                lexical_score = 0.80 * lexical_score + 0.20 * normalized_semantic
        if semantic_available is None:
            semantic_available = False

        score_percent = int(round(max(0.0, min(1.0, lexical_score)) * 100))
        if claim_type == "vague_teaser":
            status = "No evidence found"
        elif claim_type == "topic":
            status = "Supported" if lexical_score >= 0.60 and best_lexical >= 0.80 else ("Partly supported" if lexical_score >= 0.25 or best_lexical >= 0.25 else "No evidence found")
        elif claim_type == "list_count":
            status = "Supported" if best_lexical >= 0.95 else ("Partly supported" if best_lexical >= 0.25 else "No evidence found")
        elif claim_type == "free":
            status = "Supported" if best_lexical >= 0.80 else ("Partly supported" if best_lexical >= 0.25 else "No evidence found")
        elif claim_type == "quantified_outcome":
            status = "Supported" if best_lexical >= 0.95 else ("Partly supported" if best_lexical >= 0.45 else "No evidence found")
        elif claim_type == "duration":
            status = "Supported" if best_lexical >= 0.90 else ("Partly supported" if best_lexical >= 0.45 else "No evidence found")
        elif claim_type == "no_code":
            status = "Supported" if best_lexical >= 0.90 else ("Partly supported" if best_lexical >= 0.25 else "No evidence found")
        else:
            status = "Supported" if lexical_score >= 0.60 else ("Partly supported" if lexical_score >= 0.25 else "No evidence found")

        evidence = None
        matched_terms: list[str] = []
        evidence_reason = "No timestamped transcript window provided evidence for this claim."
        if best_index >= 0 and best_lexical >= 0.25:
            window = windows[best_index]
            _, matched_terms, evidence_reason = per_window[best_index]
            summary = re.sub(r"\s+", " ", str(window.get("text", ""))).strip()
            if len(summary) > 280:
                summary = summary[:277].rsplit(" ", 1)[0] + "…"
            start, end = window.get("start"), window.get("end")
            timestamp = f"[{_timestamp(start)} - {_timestamp(end)}]" if start is not None and end is not None else "Timestamp unavailable (untimed transcript)"
            evidence = {"timestamp_range": timestamp, "summary": summary, "matched_terms": matched_terms, "evidence_reason": evidence_reason}


        public_claims.append({
            "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
            "claim_type": claim_type, "status": status,
            "evidence_score_percent": score_percent,
            "matched_terms": matched_terms,
            "evidence": evidence,
            "reasoning": ("The title is too vague to define a checkable content claim. Transcript matching cannot resolve a teaser such as this." if claim_type == "vague_teaser" else (evidence_reason if status != "No evidence found" else "No clear transcript support was found for this claim. This is not proof that the claim is false; the caption transcript may be incomplete.")),
        })

    total = len(public_claims)
    supported = sum(1 for claim in public_claims if claim["status"] == "Supported")
    partly = sum(1 for claim in public_claims if claim["status"] == "Partly supported")
    no_evidence = total - supported - partly
    with_evidence = supported + partly
    timed_count = sum(1 for segment in segments if segment.get("start") is not None)
    limited = word_count < 250 or (segments and timed_count < 2)
    if word_count < 250:
        limitation_note = f"Limited transcript evidence ({word_count} words). A missing claim match may reflect incomplete captions or a short excerpt, not a false title claim."
    elif timed_count == 0:
        limitation_note = "Transcript text has no timestamps. Claim matching is possible, but the video player cannot jump to evidence times."
    else:
        limitation_note = "Results cover only the transcript available to this audit. Captions can omit visuals, demonstrations, or sections; no claim is independently fact-checked."
    return {
        "claims": public_claims,
        "total_claims": total,
        "supported_count": supported,
        "partly_supported_count": partly,
        "no_evidence_count": no_evidence,
        "claims_with_some_evidence_count": with_evidence,
        "evidence_coverage_percent": int(round(100 * with_evidence / max(1, total))),
        "fully_supported_percent": int(round(100 * supported / max(1, total))),
        "transcript_word_count": word_count,
        "timed_segment_count": timed_count,
        "limited_evidence": bool(limited),
        "limitation_note": limitation_note,
        "method": "Rule-based title-claim decomposition + transcript-window evidence matching; optional local sentence embeddings rank topic similarity, not entailment.",
        "semantic_model_used": bool(semantic_available),
        "semantic_model_name": os.getenv("VERICLICK_PROMISE_MODEL", "sentence-transformers/all-MiniLM-L6-v2") if semantic_available else None,
        "llm_used": False,
        "score_definition": "Evidence coverage is the percentage of extracted title claims with at least partial transcript support. It is not factual accuracy, truth probability, or proof that unsupported claims are false.",
    }

def build_youtube_promise_response(video_id: str, title: str, segments: list[dict[str, Any]], source: str, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    urls = youtube_urls(video_id)
    transcript = " ".join(str(segment.get("text", "")) for segment in segments).strip()
    analysis = analyze_title_promise(title, segments)
    promiseproof = analyze_title_claims(title, segments)
        # Keep overall status consistent with claim-wise evidence.
    total_claims = int(promiseproof.get("total_claims", 0) or 0)
    supported = int(promiseproof.get("supported_count", 0) or 0)
    partly = int(promiseproof.get("partly_supported_count", 0) or 0)
    missing = int(promiseproof.get("no_evidence_count", 0) or 0)

    if total_claims > 0:
        if supported == total_claims:
            analysis["fulfillment_status"] = "Fulfilled"
            analysis["reasoning"] = (
                "All extracted title claims have transcript evidence. "
                "This is transcript support, not factual verification."
            )
        elif supported + partly > 0:
            analysis["fulfillment_status"] = "Partly Fulfilled"
            analysis["reasoning"] = (
                f"Transcript evidence was found for {supported + partly} "
                f"of {total_claims} claims. The full title promise is not "
                "established by the available transcript."
            )
        else:
            analysis["fulfillment_status"] = "Not Enough Evidence"
            analysis["reasoning"] = (
                "No meaningful transcript evidence was found for the "
                "extracted title claims. Captions may be incomplete."
            )
    evidence = analysis.get("evidence_timestamps") or []
    return {
        "video_id": video_id,
        "watch_url": urls["watch_url"],
        "embed_url": urls["embed_url"],
        "thumbnail_url": (metadata or {}).get("thumbnail_url") or urls["thumbnail_url"],
        "title": title,
        "author_name": (metadata or {}).get("author_name", ""),
        "transcript_source": source,
        "transcript_word_count": len(_tokens(transcript)),
        "transcript": transcript[:MAX_TRANSCRIPT_CHARS],
        "transcript_segments": len(segments),
        "title_promise_analysis": analysis,
        "promiseproof_analysis": promiseproof,
        "disclaimer": "Automated transcript-based promise matching can miss nuance, captions, or context. It does not independently verify factual truth, intent, or overall video quality. Public YouTube captions may be unavailable or inaccurate.",
    }
