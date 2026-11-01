from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "history.sqlite3"


def db_path() -> Path:
    return Path(os.environ.get("VERICLICK_DB", str(DEFAULT_DB))).expanduser().resolve()


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                label TEXT NOT NULL,
                model_mode TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC)")


def save_analysis(title: str, payload: dict[str, Any]) -> dict[str, Any]:
    initialize_db()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO analyses(created_at, title, risk_score, label, model_mode, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            (created_at, title[:500], int(payload["risk_score"]), str(payload["label"]), str(payload["model_mode"]), json.dumps(payload, ensure_ascii=False)),
        )
        row_id = int(cursor.lastrowid)
    return {"id": row_id, "created_at": created_at, **payload, "title": title[:500]}


def list_analyses(limit: int = 50) -> list[dict[str, Any]]:
    initialize_db()
    limit = max(1, min(int(limit), 200))
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, title, payload_json FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, json.JSONDecodeError):
            payload = {}
        results.append({"id": row["id"], "created_at": row["created_at"], "title": row["title"], **payload})
    return results


def dashboard_stats() -> dict[str, Any]:
    initialize_db()
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS total, AVG(risk_score) AS average FROM analyses").fetchone()
        high = conn.execute("SELECT COUNT(*) AS n FROM analyses WHERE risk_score >= 62").fetchone()["n"]
        review = conn.execute("SELECT COUNT(*) AS n FROM analyses WHERE risk_score >= 35 AND risk_score < 62").fetchone()["n"]
        low = conn.execute("SELECT COUNT(*) AS n FROM analyses WHERE risk_score < 35").fetchone()["n"]
    return {
        "total_analyses": int(row["total"] or 0),
        "average_risk": round(float(row["average"] or 0), 1),
        "high_risk": int(high),
        "review": int(review),
        "low_risk": int(low),
    }


def clear_history() -> int:
    initialize_db()
    with _connect() as conn:
        cursor = conn.execute("DELETE FROM analyses")
        return int(cursor.rowcount)
