#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  python3.11 -m venv .venv
fi
if ! .venv/bin/python -m pip show fastapi >/dev/null 2>&1; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m uvicorn vericlick.api:app --host 127.0.0.1 --port 8000
