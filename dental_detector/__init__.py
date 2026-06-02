"""dental_detector — tooth detection in maxilla and mandible X-ray images."""

from .detector import DentalDetector
from .models import LABEL_COLORS, TOOTH_CLASSES, Detection, DetectionResult
from .utils import letterbox, load_image, to_pil
from .visualization import annotate

__version__ = "0.1.1"
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
