from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment,misc]

from .models import Detection, DetectionResult, TOOTH_CLASSES
from .utils import ImageInput, load_image, to_pil
from .visualization import annotate as _annotate


class DentalDetector:
    """Detect teeth in dental X-ray or photograph images using a YOLOv8 model.

    Supported jaw views: maxilla (upper) and mandible (lower).

    Args:
        model_path: Path to a YOLOv8 ``.pt`` weights file.
        conf_threshold: Default minimum confidence score (0–1). Detections
            below this value are discarded. Can be overridden per call.
        device: Inference device string understood by PyTorch, e.g. ``"cpu"``,
            ``"cuda"``, ``"cuda:0"``. ``None`` auto-selects GPU when available.

    Example::

        from dental_detector import DentalDetector

        detector = DentalDetector("best.pt")
        result = detector.detect("xray.jpg")
        for tooth in result:
            print(tooth.label, tooth.confidence, tooth.bbox)

        annotated = detector.annotate("xray.jpg")
        annotated.save("output.jpg")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.6,
        device: Optional[str] = None,
    ) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required: pip install ultralytics")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self._model = YOLO(str(model_path))
        if device is not None:
            self._model.to(device)

        self.conf_threshold = conf_threshold
        self.model_path = model_path

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run tooth detection on *image*.

        Args:
            image: File path, PIL Image, or numpy array (RGB or BGR).
            conf_threshold: Override the instance-level threshold for this call.

        Returns:
            :class:`DetectionResult` containing all detections above threshold.
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        bgr = load_image(image)
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw = self._model(rgb, verbose=False)

        detections = []
        for r in raw:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            names = r.names

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, classes):
                if conf < threshold:
                    continue
                cid = int(cls_id)
                label = names.get(cid, TOOTH_CLASSES.get(cid, str(cid)))
                detections.append(
                    Detection(
                        label=label,
                        class_id=cid,
                        confidence=float(conf),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        return DetectionResult(detections=detections, image_width=w, image_height=h)

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def annotate(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
        show_confidence: bool = True,
    ) -> Image.Image:
        """Detect and draw bounding boxes, returning a PIL Image.

        Args:
            image: Source image.
            conf_threshold: Minimum confidence (uses instance default if None).
            show_confidence: Whether to include confidence scores in labels.

        Returns:
            PIL RGB Image with bounding boxes overlaid.
        """
        bgr = load_image(image)
        result = self.detect(image, conf_threshold=conf_threshold)
        annotated_bgr = _annotate(bgr, result, show_confidence=show_confidence)
        return to_pil(annotated_bgr)

    def annotate_and_save(
        self,
        image: ImageInput,
        output_path: Union[str, Path],
        conf_threshold: Optional[float] = None,
        show_confidence: bool = True,
    ) -> Path:
        """Detect, annotate, and save result to *output_path*.

        Returns the resolved output path.
        """
        pil = self.annotate(image, conf_threshold=conf_threshold, show_confidence=show_confidence)
        out = Path(output_path)
        pil.save(out)
        return out

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def detect_to_json(
        self,
        image: ImageInput,
        conf_threshold: Optional[float] = None,
        indent: int = 2,
    ) -> str:
        """Return detection results as a JSON string."""
        result = self.detect(image, conf_threshold=conf_threshold)
        return json.dumps(result.to_dict(), indent=indent)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DentalDetector(model='{self.model_path.name}', "
            f"conf_threshold={self.conf_threshold})"
        )
