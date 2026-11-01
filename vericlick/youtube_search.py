"""Server-side search adapter for the official YouTube Data API v3.

API keys must be supplied by the local environment and are never returned to the
browser. Search results are kept in the current response only, not in local history.
"""
from __future__ import annotations

from typing import Any

import requests

YOUTUBE_SEARCH_ENDPOINT = "https://www.googleapis.com/youtube/v3/search"


class YouTubeSearchError(RuntimeError):
    """A friendly, user-displayable YouTube search error."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def search_youtube_videos(
    query: str,
    api_key: str,
    *,
    max_results: int = 12,
    page_token: str | None = None,
    order: str = "relevance",
    region_code: str = "IN",
    http_get=None,
) -> dict[str, Any]:
    """Search for embeddable public videos and normalize result fields.

    This uses YouTube's official search.list endpoint. It does not download video
    media or transcripts. Caption availability is handled by the Title Promise
    workflow separately.
    """
    clean_query = " ".join((query or "").split())
    if len(clean_query) < 2:
        raise YouTubeSearchError("Enter at least two characters to search YouTube.", 422)
    if len(clean_query) > 100:
        raise YouTubeSearchError("Search query must be 100 characters or fewer.", 422)
    if not api_key or not api_key.strip():
        raise YouTubeSearchError("YouTube search is not configured. Add YOUTUBE_API_KEY to your local .env file and restart Veri-Click.", 503)
    if not 1 <= int(max_results) <= 50:
        raise YouTubeSearchError("max_results must be between 1 and 50.", 422)
    allowed_orders = {"relevance", "date", "viewCount", "rating", "title"}
    if order not in allowed_orders:
        raise YouTubeSearchError("Unsupported sort order. Choose relevance, date, viewCount, rating, or title.", 422)
    if page_token and len(page_token) > 512:
        raise YouTubeSearchError("Invalid YouTube page token.", 422)

    params: dict[str, str | int] = {
        "part": "snippet",
        "q": clean_query,
        "type": "video",
        "maxResults": int(max_results),
        "order": order,
        "videoEmbeddable": "true",
        "safeSearch": "moderate",
        "regionCode": region_code,
        "key": api_key.strip(),
    }
    if page_token:
        params["pageToken"] = page_token

    getter = http_get or requests.get
    try:
        response = getter(YOUTUBE_SEARCH_ENDPOINT, params=params, timeout=(3.05, 12.0))
    except requests.RequestException as exc:
        raise YouTubeSearchError("Could not connect to YouTube Data API. Check your internet connection and try again.", 502) from exc

    try:
        payload = response.json()
    except (ValueError, AttributeError):
        payload = {}

    if getattr(response, "status_code", 200) >= 400:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = str(error.get("message") or "YouTube Data API request failed.")
        reasons = {
            str(item.get("reason", ""))
            for item in (error.get("errors") or [])
            if isinstance(item, dict)
        }
        if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}:
            raise YouTubeSearchError("YouTube search quota is exhausted for now. Check your Google Cloud quota or try again after it resets.", 429)
        if reasons & {"keyInvalid", "API_KEY_INVALID", "accessNotConfigured", "serviceDisabled"}:
            raise YouTubeSearchError("The YouTube API key is invalid or YouTube Data API v3 is not enabled for its Google Cloud project. Check your local .env and API settings.", 503)
        raise YouTubeSearchError(f"YouTube Data API rejected the request ({getattr(response, 'status_code', 'unknown')}): {message}", 502)

    items: list[dict[str, Any]] = []
    for raw in payload.get("items", []) if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        identity = raw.get("id", {})
        snippet = raw.get("snippet", {})
        video_id = identity.get("videoId") if isinstance(identity, dict) else None
        if not isinstance(video_id, str) or len(video_id) != 11:
            continue
        thumbnails = snippet.get("thumbnails", {}) if isinstance(snippet, dict) else {}
        if not isinstance(thumbnails, dict):
            thumbnails = {}
        thumbnail = next(
            (thumbnails.get(size, {}).get("url") for size in ("high", "medium", "default")
             if isinstance(thumbnails.get(size), dict) and thumbnails.get(size, {}).get("url")),
            f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        )
        title = str(snippet.get("title") or "Untitled video")
        channel_title = str(snippet.get("channelTitle") or "Unknown channel")
        items.append({
            "video_id": video_id,
            "title": title,
            "description": str(snippet.get("description") or "")[:500],
            "channel_title": channel_title,
            "channel_id": str(snippet.get("channelId") or ""),
            "published_at": str(snippet.get("publishedAt") or ""),
            "thumbnail_url": str(thumbnail),
            "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            "embed_url": f"https://www.youtube-nocookie.com/embed/{video_id}?rel=0",
        })

    page_info = payload.get("pageInfo", {}) if isinstance(payload, dict) else {}
    return {
        "query": clean_query,
        "items": items,
        "result_count": int(page_info.get("resultsPerPage", len(items)) or len(items)),
        "total_results_estimate": int(page_info.get("totalResults", 0) or 0),
        "next_page_token": payload.get("nextPageToken") if isinstance(payload, dict) else None,
        "order": order,
        "region_code": region_code,
        "provider": "YouTube Data API v3",
        "note": "Search results are live API results and are not proof of content quality or truthfulness.",
    }
