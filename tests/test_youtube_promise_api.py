import pytest
from fastapi.testclient import TestClient

from vericlick.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VERICLICK_DB", str(tmp_path / "youtube-test.sqlite3"))
    with TestClient(app) as test_client:
        yield test_client


def test_youtube_metadata_rejects_non_youtube_url(client):
    response = client.get("/api/youtube/metadata", params={"video_url": "https://evil.example/watch?v=abcdefghijk"})
    assert response.status_code == 400


def test_youtube_metadata_returns_embed_url_even_when_oembed_unavailable(client, monkeypatch):
    from vericlick import api as api_module
    monkeypatch.setattr(api_module, "fetch_youtube_metadata", lambda video_id: {"title": "Example title", "author_name": "Sample channel"})
    response = client.get("/api/youtube/metadata", params={"video_url": "https://youtu.be/abcdefghijk"})
    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "abcdefghijk"
    assert body["title"] == "Example title"
    assert body["embed_url"].endswith("/abcdefghijk")


def test_youtube_promise_check_with_pasted_timed_transcript(client, monkeypatch):
    from vericlick import api as api_module
    monkeypatch.setattr(api_module, "fetch_youtube_metadata", lambda video_id: {"title": "What is deep learning?", "author_name": "Demo"})
    response = client.post("/api/youtube/promise-check", json={
        "video_url": "https://www.youtube.com/watch?v=abcdefghijk",
        "title": "What is deep learning?",
        "transcript": "[00:00 - 00:20] What is deep learning? Deep learning is a subset of machine learning inspired by the human brain.\n[00:20 - 00:40] It uses layers of neural networks to learn patterns from examples.",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["title_promise_analysis"]["fulfillment_status"] == "Fulfilled"
    assert body["title_promise_analysis"]["evidence_timestamps"]
    assert body["transcript_source"].startswith("user-pasted")
    assert body["promiseproof_analysis"]["total_claims"] >= 1
    assert "evidence_coverage_percent" in body["promiseproof_analysis"]
    assert body["clickbait_assessment"]["risk_score"] is not None
    assert "secret" not in response.text.lower()


def test_youtube_promise_check_falls_back_to_public_transcript_fetch(client, monkeypatch):
    from vericlick import api as api_module
    monkeypatch.setattr(api_module, "fetch_youtube_metadata", lambda video_id: {"title": "How to learn Python", "author_name": "Demo"})
    monkeypatch.setattr(api_module, "fetch_youtube_transcript", lambda video_id: [
        {"start": 0, "end": 15, "text": "This Python tutorial teaches you how to write your first program."},
        {"start": 15, "end": 30, "text": "First, define a variable, then run the program and check the result."},
    ])
    response = client.post("/api/youtube/promise-check", json={"video_url": "abcdefghijk"})
    assert response.status_code == 200
    assert response.json()["transcript_source"] == "public YouTube captions"
    assert response.json()["title"] == "How to learn Python"


def test_youtube_promise_check_explains_missing_captions(client, monkeypatch):
    from vericlick import api as api_module
    monkeypatch.setattr(api_module, "fetch_youtube_metadata", lambda video_id: {"title": "How to learn Python"})
    monkeypatch.setattr(api_module, "fetch_youtube_transcript", lambda video_id: (_ for _ in ()).throw(RuntimeError("Public captions could not be retrieved. Paste transcript.")))
    response = client.post("/api/youtube/promise-check", json={"video_url": "abcdefghijk"})
    assert response.status_code == 422
    assert "captions" in response.json()["detail"].lower()


def test_youtube_search_status_does_not_expose_key(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "secret-test-key")
    response = client.get("/api/youtube/search-status")
    assert response.status_code == 200
    assert response.json()["configured"] is True
    assert "secret-test-key" not in response.text


def test_youtube_search_requires_local_api_key(client, monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    response = client.get("/api/youtube/search", params={"q": "deep learning"})
    assert response.status_code == 503
    assert "YOUTUBE_API_KEY" in response.json()["detail"]


def test_youtube_search_returns_normalized_cards(client, monkeypatch):
    from vericlick import api as api_module
    monkeypatch.setenv("YOUTUBE_API_KEY", "secret-test-key")
    monkeypatch.setattr(api_module, "search_youtube_videos", lambda q, key, **kwargs: {
        "query": q, "items": [{"video_id": "abcdefghijk", "title": "Deep Learning", "channel_title": "Example", "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg", "watch_url": "https://www.youtube.com/watch?v=abcdefghijk", "embed_url": "https://www.youtube-nocookie.com/embed/abcdefghijk?rel=0"}],
        "result_count": 1, "total_results_estimate": 1, "next_page_token": None, "order": kwargs["order"], "region_code": "IN", "provider": "YouTube Data API v3", "note": "Live API results"
    })
    response = client.get("/api/youtube/search", params={"q": "deep learning", "max_results": 12})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["items"][0]["video_id"] == "abcdefghijk"
    assert "secret-test-key" not in response.text


def test_youtube_search_status_treats_example_placeholder_as_unconfigured(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "replace_with_your_youtube_data_api_v3_key")
    response = client.get("/api/youtube/search-status")
    assert response.status_code == 200
    assert response.json()["configured"] is False
