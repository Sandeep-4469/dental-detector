# dental-detector

[![PyPI version](https://img.shields.io/pypi/v/dental-detector.svg)](https://pypi.org/project/dental-detector/)
[![Python](https://img.shields.io/pypi/pyversions/dental-detector.svg)](https://pypi.org/project/dental-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/sandeepvissa/dental-detector/actions/workflows/tests.yml/badge.svg)](https://github.com/sandeepvissa/dental-detector/actions/workflows/tests.yml)

**Tooth detection in maxilla and mandible dental images using YOLOv8.**

`dental-detector` is a lightweight, open-source Python package that wraps a YOLOv8 model trained to locate and classify teeth in dental radiographs and photographs. It detects 7 tooth types across both arches and ships with a clean Python API and a CLI.

## Detected classes

| Class ID | Tooth |
|----------|-------|
| 0 | 1st Molar |
| 1 | 1st Premolar |
| 2 | 2nd Molar |
| 3 | 2nd Premolar |
| 4 | Canine |
| 5 | Central Incisor |
| 6 | Lateral Incisor |

## Installation

```bash
pip install dental-detector
```

> **Note:** You must supply your own trained YOLOv8 weights (`.pt` file).
> The package does not bundle a model; it provides the inference wrapper.

## Quick start

```python
from dental_detector import DentalDetector

detector = DentalDetector("best.pt")

# --- Detect ---
result = detector.detect("xray_maxilla.jpg")
print(f"Found {len(result)} teeth")
for tooth in result:
    print(tooth.label, tooth.confidence, tooth.bbox)

# --- Annotate and save ---
annotated = detector.annotate("xray_maxilla.jpg")   # PIL Image
annotated.save("annotated.jpg")

# or in one step:
detector.annotate_and_save("xray_maxilla.jpg", "annotated.jpg")

# --- Export as JSON ---
json_str = detector.detect_to_json("xray_mandible.jpg")
print(json_str)
```

### Working with PIL / numpy directly

```python
from PIL import Image
import numpy as np

pil_img = Image.open("xray.jpg")
result = detector.detect(pil_img)

arr = np.array(pil_img)         # RGB numpy array also works
result = detector.detect(arr)
```

### Confidence threshold

```python
# Set at construction time
detector = DentalDetector("best.pt", conf_threshold=0.7)

# Or override per call
result = detector.detect("xray.jpg", conf_threshold=0.5)
```

### Filtering results

```python
# Keep only molars
molars = result.filter_by_label("1st Molar")

# Keep high-confidence detections
high_conf = result.filter_by_confidence(0.85)

# Unique tooth types present in the image
print(result.unique_labels)
```

## Command-line interface

```bash
# Print detected teeth
dental-detector xray.jpg --model best.pt

# Save annotated image
dental-detector xray.jpg --model best.pt --output annotated.jpg

# Output as JSON
dental-detector xray.jpg --model best.pt --json

# Set confidence threshold
dental-detector xray.jpg --model best.pt --conf 0.7
```

## API reference

### `DentalDetector`

| Method | Returns | Description |
|--------|---------|-------------|
| `detect(image, conf_threshold=None)` | `DetectionResult` | Run detection |
| `annotate(image, ...)` | `PIL.Image` | Detect + draw boxes |
| `annotate_and_save(image, path, ...)` | `Path` | Detect, draw, save |
| `detect_to_json(image, ...)` | `str` | Detections as JSON |

### `DetectionResult`

| Attribute / Method | Description |
|--------------------|-------------|
| `detections` | `list[Detection]` |
| `labels` | All label strings |
| `unique_labels` | Sorted unique labels |
| `filter_by_label(label)` | Returns filtered `DetectionResult` |
| `filter_by_confidence(threshold)` | Returns filtered `DetectionResult` |
| `to_dict()` | Serializable dict |

### `Detection`

| Attribute | Type | Description |
|-----------|------|-------------|
| `label` | `str` | Tooth class name |
| `class_id` | `int` | Class index (0–6) |
| `confidence` | `float` | Model confidence (0–1) |
| `bbox` | `tuple[float, float, float, float]` | `(x1, y1, x2, y2)` pixels |
| `center` | `tuple[float, float]` | Bounding box center |
| `width` / `height` / `area` | `float` | Bounding box dimensions |

## Development

```bash
git clone https://github.com/sandeepvissa/dental-detector
cd dental-detector
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check dental_detector tests
```

## License

MIT — see [LICENSE](LICENSE).
