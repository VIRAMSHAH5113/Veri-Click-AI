from __future__ import annotations

from types import SimpleNamespace

import pytest

from vericlick.youtube_search import YouTubeSearchError, search_youtube_videos


def fake_response(payload, status_code=200):
    return SimpleNamespace(status_code=status_code, json=lambda: payload)


def test_search_normalizes_results_and_next_page_token():
    captured = {}
    def fake_get(url, *, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return fake_response({
            "nextPageToken": "next-page",
            "pageInfo": {"totalResults": 124, "resultsPerPage": 1},
            "items": [{
                "id": {"kind": "youtube#video", "videoId": "abcdefghijk"},
                "snippet": {
                    "title": "Deep Learning Basics", "description": "A tutorial",
                    "channelTitle": "Example Channel", "channelId": "UCsample",
                    "publishedAt": "2025-01-02T03:04:05Z",
                    "thumbnails": {"high": {"url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"}},
                },
            }]
        })

    result = search_youtube_videos("  deep   learning ", "test-secret", max_results=12, page_token="old-page", order="date", http_get=fake_get)
    assert result["query"] == "deep learning"
    assert result["items"][0]["video_id"] == "abcdefghijk"
    assert result["items"][0]["title"] == "Deep Learning Basics"
    assert result["items"][0]["embed_url"].endswith("/abcdefghijk?rel=0")
    assert result["next_page_token"] == "next-page"
    assert captured["params"]["key"] == "test-secret"
    assert captured["params"]["videoEmbeddable"] == "true"
    assert captured["params"]["pageToken"] == "old-page"


def test_search_requires_key():
    with pytest.raises(YouTubeSearchError, match="YOUTUBE_API_KEY") as err:
        search_youtube_videos("deep learning", "")
    assert err.value.status_code == 503


def test_search_validates_query_and_order():
    with pytest.raises(YouTubeSearchError):
        search_youtube_videos("x", "key")
    with pytest.raises(YouTubeSearchError):
        search_youtube_videos("deep learning", "key", order="weird")


def test_search_translates_quota_errors():
    response = fake_response({"error": {"message": "Quota exceeded", "errors": [{"reason": "quotaExceeded"}]}}, 403)
    with pytest.raises(YouTubeSearchError, match="quota is exhausted") as err:
        search_youtube_videos("deep learning", "key", http_get=lambda *a, **kw: response)
    assert err.value.status_code == 429
