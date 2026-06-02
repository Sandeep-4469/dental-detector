from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    """Single tooth detection result."""

    label: str
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": {
                "x1": round(self.bbox[0], 2),
                "y1": round(self.bbox[1], 2),
                "x2": round(self.bbox[2], 2),
                "y2": round(self.bbox[3], 2),
            },
        }


@dataclass
class DetectionResult:
    """Full detection output for one image."""

    detections: List[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def labels(self) -> List[str]:
        return [d.label for d in self.detections]

    @property
    def unique_labels(self) -> List[str]:
        return sorted(set(self.labels))

    def filter_by_label(self, label: str) -> "DetectionResult":
        return DetectionResult(
            detections=[d for d in self.detections if d.label == label],
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        return DetectionResult(
            detections=[d for d in self.detections if d.confidence >= threshold],
            image_width=self.image_width,
            image_height=self.image_height,
        )

    def to_dict(self) -> dict:
        return {
            "image_size": {"width": self.image_width, "height": self.image_height},
            "count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)


# Canonical tooth class names as trained
TOOTH_CLASSES = {
    0: "1st Molar",
    1: "1st Premolar",
    2: "2nd Molar",
    3: "2nd Premolar",
    4: "Canine",
    5: "Central Incisor",
    6: "Lateral Incisor",
}

# FDI-style color map for visualization
LABEL_COLORS = {
    "1st Molar": (255, 87, 51),
    "2nd Molar": (255, 140, 0),
    "1st Premolar": (50, 205, 50),
    "2nd Premolar": (0, 180, 100),
    "Canine": (30, 144, 255),
    "Central Incisor": (148, 0, 211),
    "Lateral Incisor": (255, 20, 147),
}
