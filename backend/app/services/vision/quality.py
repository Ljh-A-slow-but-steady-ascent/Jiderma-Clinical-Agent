from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def assess_image_quality(image_path: Path) -> tuple[float, dict[str, float]]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        brightness_mean = float(gray.mean()) / 255.0
        brightness = _clamp(1.0 - abs(brightness_mean - 0.5) / 0.5)

        edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        blur = _clamp(float(edges.var()) / 1200.0)

        width, height = image.size
        resolution = _clamp(min(width, height) / 512.0)

    components = {"brightness": brightness, "blur": blur, "resolution": resolution}
    quality = 0.35 * brightness + 0.40 * blur + 0.25 * resolution
    return _clamp(quality), components
