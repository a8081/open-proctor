from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

FONTS = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
]


def _find_font() -> Path | None:
    for f in FONTS:
        if f.exists():
            return f
    return None


def _draw_text(size=(320, 120), text: str = "Hello World") -> np.ndarray:
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_path = _find_font()
    font = ImageFont.truetype(str(font_path), 28) if font_path else None
    draw.text((10, 40), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.full((100, 200, 3), 255, dtype=np.uint8)


@pytest.fixture
def keyword_frame() -> np.ndarray:
    return _draw_text(text="Join our discord server for help")


@pytest.fixture
def clean_frame() -> np.ndarray:
    return _draw_text(text="The capital of France is Paris")


@pytest.fixture
def multi_keyword_frame() -> np.ndarray:
    return _draw_text(text="Check ChatGPT and Discord for answers")


@pytest.fixture
def tiny_frame() -> np.ndarray:
    return np.full((10, 10, 3), 128, dtype=np.uint8)


@pytest.fixture
def tmp_frames_dir(tmp_path) -> Path:
    d = tmp_path / "frames"
    d.mkdir()
    for i in range(5):
        img = _draw_text(text=f"Frame {i} content")
        cv2.imwrite(str(d / f"frame_{i:04d}.jpg"), img)
    return d


@pytest.fixture
def tmp_suspects_dir(tmp_path) -> Path:
    d = tmp_path / "suspects"
    d.mkdir()
    return d
