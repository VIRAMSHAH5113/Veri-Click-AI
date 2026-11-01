from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from PIL import Image, UnidentifiedImageError

from . import __version__
from .analyzer import analyze, model_status
from .storage import clear_history, dashboard_stats, list_analyses, save_analysis
from .media_analyzer import (
    ALLOWED_MEDIA_SUFFIXES, MAX_MEDIA_BYTES, MediaDependencyError,
    MediaProcessingError, analyze_media_file,
)
from .youtube_search import YouTubeSearchError, search_youtube_videos
from .title_promise import (
    extract_youtube_video_id, fetch_youtube_metadata, fetch_youtube_transcript,
    parse_timestamped_transcript, youtube_urls, build_youtube_promise_response,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"

# The key remains server-side. Copy .env.example to .env and add the key locally.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # Keep the core app usable before optional dotenv is installed.
    pass


def _youtube_api_key() -> str:
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    placeholders = {
        "replace_with_your_youtube_data_api_v3_key",
        "your_actual_key_here",
        "your_api_key",
        "changeme",
    }
    if not key or key.lower() in placeholders or key.lower().startswith("replace_with_"):
        return ""
    return key


MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_THUMBNAIL_HOSTS = ("ytimg.com", "youtube.com", "googleusercontent.com")

app = FastAPI(
    title="Veri-Click AI API",
    description="Local-first multimodal clickbait research prototype; demo risk index is not a truth verdict.",
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "https://www.youtube.com", "https://youtube.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AnalyzeRequest(BaseModel):
    title: str = Field(min_length=3, max_length=500)
    image_data: str | None = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 100, description="Optional image data URL or base64-encoded bytes")
    thumbnail_url: str | None = Field(default=None, max_length=2048, description="Optional YouTube thumbnail URL; restricted to public thumbnail hosts")
    save_history: bool = True

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Title must contain at least 3 non-space characters.")
        return value


class YouTubePromiseRequest(BaseModel):
    video_url: str = Field(min_length=3, max_length=2048)
    title: str = Field(default="", max_length=500)
    transcript: str = Field(default="", max_length=40000)


class BatchItem(BaseModel):
    title: str = Field(min_length=3, max_length=500)
    image_data: str | None = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 100)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Title must contain at least 3 non-space characters.")
        return value


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=200)


def decode_image_data(data: str | None) -> bytes | None:
    if not data:
        return None
    encoded = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    if len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 100):
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MB upload limit.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Image upload is not valid base64 data.") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MB upload limit.")
    try:
        with Image.open(BytesIO(raw)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a supported image.") from exc
    return raw


def _safe_thumbnail_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Only HTTPS thumbnail URLs are accepted.")
    hostname = parsed.hostname.lower().rstrip(".")
    if not any(hostname == host or hostname.endswith("." + host) for host in ALLOWED_THUMBNAIL_HOSTS):
        raise HTTPException(status_code=400, detail="Thumbnail URL host is not allow-listed.")
    if parsed.port not in (None, 443) or parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Invalid thumbnail URL authority.")
    return value


def fetch_thumbnail(value: str | None) -> bytes | None:
    if not value:
        return None
    safe_url = _safe_thumbnail_url(value)
    try:
        with requests.get(safe_url, timeout=(2.5, 4.0), stream=True, allow_redirects=False, headers={"User-Agent": "Veri-Click-AI/1.0"}) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Thumbnail URL did not return an image.")
            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Remote thumbnail exceeds the 5 MB limit.")
                chunks.append(chunk)
            raw = b"".join(chunks)
        with Image.open(BytesIO(raw)) as im:
            im.verify()
        return raw
    except HTTPException:
        raise
    except (requests.RequestException, UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.warning("Thumbnail fetch failed: %s", exc)
        raise HTTPException(status_code=400, detail="Could not retrieve a valid thumbnail from that URL.") from exc


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, **model_status()}


@app.get("/api/dashboard")
def dashboard():
    return {**dashboard_stats(), "model": model_status()}


@app.get("/api/history")
def history(limit: int = 50):
    return {"items": list_analyses(limit=limit)}


@app.delete("/api/history")
def delete_history():
    count = clear_history()
    return {"deleted": count}


@app.get("/api/model-details")
def model_details():
    import json
    artifacts_dir = ROOT / "artifacts"
    names = [
        "multimodal_model.pt", "vocab.json", "model_config.json", "test_metrics.json",
        "model_comparison_metrics.json", "model_comparison.csv",
        "comparison_models/bigru_text.pt", "comparison_models/cnn_image.pt",
        "comparison_models/mlp_features.pt", "comparison_models/multimodal_fusion.pt",
    ]
    artifacts = [{"name": name, "exists": (artifacts_dir / name).is_file()} for name in names]
    metrics = None
    metrics_path = artifacts_dir / "test_metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = None
    comparison = None
    comparison_path = artifacts_dir / "model_comparison_metrics.json"
    if comparison_path.is_file():
        try:
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            comparison = None
    return {"artifacts": artifacts, "test_metrics": metrics, "comparison": comparison, "model": model_status()}


@app.get("/api/model-comparison.csv", include_in_schema=False)
def download_model_comparison():
    path = ROOT / "artifacts" / "model_comparison.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Train the four-model comparison first; no result CSV exists yet.")
    return FileResponse(path, filename="vericlick_model_comparison.csv", media_type="text/csv")


@app.get("/api/demo-cases")
def demo_cases():
    import json
    path = ROOT / "data" / "demo_cases.json"
    return {"items": json.loads(path.read_text(encoding="utf-8"))}


@app.get("/api/media-capabilities")
def media_capabilities():
    import importlib.util
    import os
    installed = importlib.util.find_spec("faster_whisper") is not None
    return {
        "speech_ai_installed": installed,
        "model": os.getenv("VERICLICK_WHISPER_MODEL", "tiny"),
        "max_upload_mb": MAX_MEDIA_BYTES // (1024 * 1024),
        "message": "Ready for local speech transcription" if installed else "Install optional speech dependencies with pip install -r requirements-media.txt",
    }


@app.post("/api/media-analyze")
async def analyze_uploaded_media(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    goal: str = Form(default=""),
):
    """Analyze an uploaded audio/video file, without saving the media or transcript."""
    filename = file.filename or "uploaded-media"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_MEDIA_SUFFIXES:
        await file.close()
        raise HTTPException(status_code=415, detail="Unsupported media extension. Use MP4, MOV, MKV, WebM, AVI, MP3, WAV, M4A, AAC, OGG, FLAC, or WMA.")

    temp_path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(prefix="vericlick-media-", suffix=suffix, delete=False) as destination:
            temp_path = Path(destination.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_MEDIA_BYTES:
                    raise HTTPException(status_code=413, detail=f"Media file exceeds the {MAX_MEDIA_BYTES // (1024 * 1024)} MB upload limit. Try a shorter clip.")
                destination.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded media file is empty.")
        result = await run_in_threadpool(analyze_media_file, temp_path, filename, title[:300], goal[:500])
        return result
    except MediaDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MediaProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary media file")


@app.get("/api/youtube/metadata")
def youtube_metadata(video_url: str = Query(min_length=3, max_length=2048)):
    """Resolve and return basic metadata for an embeddable public YouTube video."""
    try:
        video_id = extract_youtube_video_id(video_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    metadata = fetch_youtube_metadata(video_id)
    return {"video_id": video_id, **youtube_urls(video_id), **metadata}


@app.get("/api/youtube/search-status")
def youtube_search_status():
    configured = bool(_youtube_api_key())
    return {
        "configured": configured,
        "provider": "YouTube Data API v3",
        "message": (
            "YouTube search is ready. Search is performed by the local server; the API key is not sent to the browser."
            if configured else
            "YouTube search needs a local API key. Copy .env.example to .env, set YOUTUBE_API_KEY, and restart the server."
        ),
    }


@app.get("/api/youtube/search")
def youtube_search(
    q: str = Query(min_length=2, max_length=100),
    max_results: int = Query(default=12, ge=1, le=50),
    page_token: str | None = Query(default=None, max_length=512),
    order: str = Query(default="relevance", pattern="^(relevance|date|viewCount|rating|title)$"),
):
    """Search embeddable public YouTube videos using the user's locally configured API key."""
    api_key = _youtube_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="YouTube search is not configured. Copy .env.example to .env, add YOUTUBE_API_KEY, and restart Veri-Click.",
        )
    try:
        return search_youtube_videos(
            q, api_key, max_results=max_results, page_token=page_token, order=order, region_code="IN"
        )
    except YouTubeSearchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/api/youtube/promise-check")
def youtube_title_promise_check(request: YouTubePromiseRequest):
    """Check whether a title promise is supported by public captions or pasted transcript."""
    try:
        video_id = extract_youtube_video_id(request.video_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Avoid an unnecessary public metadata request when the user already supplied the title.
    metadata = fetch_youtube_metadata(video_id) if not request.title.strip() else {}
    title = (request.title or metadata.get("title", "")).strip()[:500]
    if len(title) < 3:
        raise HTTPException(status_code=422, detail="Enter the video title, or click Load video to try fetching the public title automatically.")

    if request.transcript.strip():
        segments = parse_timestamped_transcript(request.transcript)
        source = "user-pasted timestamped transcript" if any(segment.get("start") is not None for segment in segments) else "user-pasted transcript (timestamps not supplied)"
    else:
        try:
            segments = fetch_youtube_transcript(video_id)
        except RuntimeError as exc:
            message = str(exc)
            code = 503 if "not installed" in message.lower() else 422
            raise HTTPException(status_code=code, detail=message) from exc
        source = "public YouTube captions"

    if not segments or not any(str(segment.get("text", "")).strip() for segment in segments):
        raise HTTPException(status_code=422, detail="No transcript text was provided or retrieved. Paste the transcript with timestamps to continue.")
    try:
        result = build_youtube_promise_response(video_id, title, segments, source, metadata)
        # Add a separate title-only clickbait-risk view so PromiseProof can bring the two
        # existing outputs together without confusing clickbait risk with factual accuracy.
        clickbait = analyze(title, None)
        result["clickbait_assessment"] = {
            "risk_score": clickbait.get("risk_score"),
            "label": clickbait.get("label"),
            "risk_band": clickbait.get("risk_band"),
            "model_mode": clickbait.get("model_mode"),
            "model_name": clickbait.get("model_name"),
            "reasons": clickbait.get("reasons", []),
            "disclaimer": clickbait.get("disclaimer", "Title-only clickbait risk estimate; not a truth verdict."),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return result


@app.post("/api/analyze")
def analyze_video(request: AnalyzeRequest):
    image_bytes = decode_image_data(request.image_data)
    if image_bytes is None and request.thumbnail_url:
        image_bytes = fetch_thumbnail(request.thumbnail_url)
    try:
        result = analyze(request.title, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["title"] = request.title
    if request.save_history:
        return save_analysis(request.title, result)
    return result


@app.post("/api/batch")
def analyze_batch(request: BatchRequest):
    results: list[dict[str, Any]] = []
    for index, item in enumerate(request.items, start=1):
        image_bytes = decode_image_data(item.image_data)
        result = analyze(item.title, image_bytes)
        result.update({"title": item.title, "row": index})
        results.append(result)
    high = sum(item["risk_score"] >= 62 for item in results)
    review = sum(35 <= item["risk_score"] < 62 for item in results)
    low = len(results) - high - review
    avg = round(sum(item["risk_score"] for item in results) / len(results), 1)
    return {"items": results, "summary": {"total": len(results), "high_risk": high, "review": review, "low_risk": low, "average_risk": avg}}
