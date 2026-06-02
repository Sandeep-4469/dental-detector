from __future__ import annotations

import os
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
from PIL import Image


ImageInput = Union[str, Path, np.ndarray, Image.Image]


def load_image(source: ImageInput) -> np.ndarray:
    """Load any supported image type into a BGR numpy array."""
    if isinstance(source, (str, Path)):
        path = str(source)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not decode image: {path}")
        return img
    if isinstance(source, Image.Image):
        return cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2BGR)
        if source.shape[2] == 3:
            # Assume RGB from PIL/matplotlib; convert to BGR for OpenCV
            return source[:, :, ::-1].copy()
        raise ValueError(f"Unsupported array shape: {source.shape}")
    raise TypeError(f"Unsupported image type: {type(source)}")


def to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def letterbox(
    image: np.ndarray,
    target: Tuple[int, int] = (640, 640),
    pad_value: int = 114,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize with aspect ratio preserved and pad to `target` (W, H).

    Returns:
        padded image, scale factor, (left_pad, top_pad)
    """
    h, w = image.shape[:2]
    tw, th = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    left, top = (tw - nw) // 2, (th - nh) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)
