"""dental_detector — tooth detection in maxilla and mandible X-ray images."""

from .detector import DentalDetector
from .models import Detection, DetectionResult, TOOTH_CLASSES, LABEL_COLORS
from .utils import load_image, to_pil, letterbox
from .visualization import annotate

__version__ = "0.1.0"
__all__ = [
    "DentalDetector",
    "Detection",
    "DetectionResult",
    "TOOTH_CLASSES",
    "LABEL_COLORS",
    "load_image",
    "to_pil",
    "letterbox",
    "annotate",
]
