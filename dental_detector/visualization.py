from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .models import Detection, DetectionResult, LABEL_COLORS


_DEFAULT_COLOR = (0, 255, 0)


def annotate(
    image: np.ndarray,
    result: DetectionResult,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image copy."""
    out = image.copy()
    for det in result.detections:
        color = LABEL_COLORS.get(det.label, _DEFAULT_COLOR)
        # BGR order for OpenCV
        bgr = (color[2], color[1], color[0])
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, line_thickness)
        label_text = f"{det.label} {det.confidence:.2f}" if show_confidence else det.label
        _draw_label(out, label_text, x1, y1, bgr, font_scale, line_thickness)
    return out


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color,
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    # Background rectangle above the bounding box
    bg_y1 = max(0, y - th - baseline - 4)
    bg_y2 = max(th + baseline + 4, y)
    cv2.rectangle(image, (x, bg_y1), (x + tw + 4, bg_y2), color, -1)
    text_color = _contrast_color(color)
    cv2.putText(
        image, text,
        (x + 2, bg_y2 - baseline - 2),
        font, font_scale, text_color, thickness, cv2.LINE_AA,
    )


def _contrast_color(bgr) -> tuple:
    """Return black or white depending on perceived luminance."""
    b, g, r = bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)
