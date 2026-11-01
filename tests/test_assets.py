from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class IdParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])


def test_frontend_contains_expected_controls_and_no_duplicate_ids():
    html_path = ROOT / "static" / "index.html"
    html = html_path.read_text(encoding="utf-8")
    parser = IdParser()
    parser.feed(html)
    assert {"video-title", "analyze-btn", "batch-file", "batch-analyze-btn", "artifact-list"}.issubset(parser.ids)
    assert len(parser.ids) == len(set(parser.ids)), "HTML IDs must be unique"
    assert "/static/styles.css" in html
    assert "/static/app.js" in html


def test_extension_manifest_references_real_assets_and_youtube_hosts():
    manifest_path = ROOT / "extension" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert any("youtube.com" in match for match in manifest["content_scripts"][0]["matches"])
    assert (manifest_path.parent / "content.js").is_file()
    assert (manifest_path.parent / "content.css").is_file()
    assert any("127.0.0.1:8000" in host for host in manifest["host_permissions"])
