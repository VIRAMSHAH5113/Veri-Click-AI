from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from vericlick.api import app


@pytest.fixture

def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VERICLICK_DB", str(tmp_path / "test-history.sqlite3"))
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_home_are_available(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert client.get("/").status_code == 200


def test_analyze_saves_history_and_returns_explanation(client):
    response = client.post("/api/analyze", json={"title": "STOP! You will not believe this shocking secret", "save_history": True})
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["risk_score"] <= 100
    assert result["label"]
    assert result["reasons"]
    assert result["id"] >= 1

    history = client.get("/api/history").json()["items"]
    assert len(history) == 1
    assert history[0]["title"] == "STOP! You will not believe this shocking secret"
    assert client.get("/api/dashboard").json()["total_analyses"] == 1


def test_analyze_with_uploaded_image(client, png_data_url):
    response = client.post("/api/analyze", json={"title": "A colorful thumbnail example", "image_data": png_data_url, "save_history": False})
    assert response.status_code == 200
    assert response.json()["image_available"] is True
    assert response.json()["visual_metrics"]


def test_invalid_image_data_is_rejected(client):
    response = client.post("/api/analyze", json={"title": "Image upload test", "image_data": "data:image/png;base64,not-valid!", "save_history": False})
    assert response.status_code == 400
    assert "base64" in response.json()["detail"].lower()


def test_invalid_or_unsafe_thumbnail_url_is_rejected_before_network(client):
    for url in ("http://127.0.0.1/private", "https://evil.example/image.jpg", "https://user:pass@i.ytimg.com/image.jpg"):
        response = client.post("/api/analyze", json={"title": "Thumbnail URL validation", "thumbnail_url": url, "save_history": False})
        assert response.status_code == 400


def test_blank_and_short_titles_are_rejected(client):
    response = client.post("/api/analyze", json={"title": "  ", "save_history": False})
    assert response.status_code == 422
    response = client.post("/api/analyze", json={"title": "a", "save_history": False})
    assert response.status_code == 422


def test_batch_analyze_and_history_clear(client):
    response = client.post("/api/batch", json={"items": [
        {"title": "STOP! This unbelievable secret changes everything"},
        {"title": "How to calculate a derivative: step by step tutorial"},
    ]})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total"] == 2
    assert len(body["items"]) == 2
    assert body["items"][0]["row"] == 1
    assert body["summary"]["average_risk"] >= 0
    assert client.get("/api/model-details").status_code == 200
    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert client.get("/api/history").json()["items"] == []


def test_batch_requires_at_least_one_item(client):
    response = client.post("/api/batch", json={"items": []})
    assert response.status_code == 422


def test_demo_cases_and_sample_assets_are_served(client):
    demos = client.get("/api/demo-cases")
    assert demos.status_code == 200
    assert len(demos.json()["items"]) >= 3
    asset = client.get("/static/samples/tutorial.jpg")
    assert asset.status_code == 200
    assert asset.headers["content-type"].startswith("image/")


def test_batch_rejects_blank_title(client):
    response = client.post("/api/batch", json={"items": [{"title": "   "}]})
    assert response.status_code == 422


def test_cors_does_not_grant_arbitrary_web_origins(client):
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_remote_thumbnail_redirects_are_not_followed(client, monkeypatch):
    from unittest.mock import MagicMock

    from vericlick import api as api_module

    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.headers = {"content-type": "text/html"}
    fake_response.status_code = 302
    fake_response.iter_content.return_value = []
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(kwargs)
        return fake_response

    monkeypatch.setattr(api_module.requests, "get", fake_get)
    response = client.post("/api/analyze", json={
        "title": "Test thumbnail redirect handling",
        "thumbnail_url": "https://i.ytimg.com/redirect.jpg",
        "save_history": False,
    })
    assert response.status_code == 400
    assert captured.get("allow_redirects") is False


def test_media_capabilities_reports_optional_speech_setup(client):
    response = client.get("/api/media-capabilities")
    assert response.status_code == 200
    body = response.json()
    assert "speech_ai_installed" in body
    assert body["max_upload_mb"] == 200


def test_media_upload_rejects_unsupported_extension(client):
    response = client.post(
        "/api/media-analyze",
        files={"file": ("notes.txt", b"not a media file", "text/plain")},
        data={"title": "Test file", "goal": "learn Python"},
    )
    assert response.status_code == 415
    assert "unsupported media" in response.json()["detail"].lower()


def test_media_upload_uses_temp_file_and_does_not_store_upload(client, monkeypatch):
    from vericlick import api as api_module
    captured = {}

    def fake_analyze_media_file(path, filename, title="", goal=""):
        captured["path"] = path
        assert path.read_bytes() == b"fake-video-content"
        return {
            "filename": filename,
            "title": title,
            "goal": goal,
            "recommendation": "Likely worth a look for your goal",
            "worth_score": 71,
            "transcript": "example transcript",
        }

    monkeypatch.setattr(api_module, "analyze_media_file", fake_analyze_media_file)
    response = client.post(
        "/api/media-analyze",
        files={"file": ("sample.mp4", b"fake-video-content", "video/mp4")},
        data={"title": "Sample lecture", "goal": "learn Python basics"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["worth_score"] == 71
    assert body["goal"] == "learn Python basics"
    assert captured["path"].exists() is False  # temporary upload is deleted after analysis
    assert client.get("/api/history").json()["items"] == []


def test_model_comparison_api_contract(client):
    details = client.get("/api/model-details")
    assert details.status_code == 200
    assert "comparison" in details.json()
    assert any(item["name"] == "model_comparison_metrics.json" for item in details.json()["artifacts"])
    # The CSV may exist if model comparison was already trained.
    download = client.get("/api/model-comparison.csv")

    if download.status_code == 404:
        assert "train" in download.json()["detail"].lower()
    else:
        assert download.status_code == 200
        assert download.text.strip()
        assert "csv" in download.headers.get("content-type", "").lower()
