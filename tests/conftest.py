from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

# Ensure the project root is importable even when pytest is launched via a global entry point.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from PIL import Image, ImageDraw


@pytest.fixture

def png_data_url():
    """Create a small deterministic synthetic thumbnail without network access."""
    image = Image.new("RGB", (320, 180), (35, 95, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 160, 180), fill=(245, 160, 20))
    draw.ellipse((110, 25, 210, 125), fill=(245, 245, 245))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture

def sample_png_bytes(png_data_url):
    return base64.b64decode(png_data_url.split(",", 1)[1])
