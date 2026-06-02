"""Tests for dental_detector package."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dental_detector import DentalDetector, Detection, DetectionResult, TOOTH_CLASSES
from dental_detector.utils import load_image, letterbox, to_pil
from dental_detector.visualization import annotate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image_rgb() -> np.ndarray:
    """100x80 white RGB image."""
    return np.full((80, 100, 3), 255, dtype=np.uint8)


@pytest.fixture
def dummy_pil_image() -> Image.Image:
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(
        label="Central Incisor",
        class_id=5,
        confidence=0.92,
        bbox=(10.0, 20.0, 80.0, 70.0),
    )


@pytest.fixture
def sample_result(sample_detection) -> DetectionResult:
    return DetectionResult(
        detections=[sample_detection],
        image_width=100,
        image_height=80,
    )


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

class TestDetection:
    def test_center(self, sample_detection):
        cx, cy = sample_detection.center
        assert cx == pytest.approx(45.0)
        assert cy == pytest.approx(45.0)

    def test_width_height(self, sample_detection):
        assert sample_detection.width == pytest.approx(70.0)
        assert sample_detection.height == pytest.approx(50.0)

    def test_area(self, sample_detection):
        assert sample_detection.area == pytest.approx(3500.0)

    def test_to_dict_keys(self, sample_detection):
        d = sample_detection.to_dict()
        assert set(d.keys()) == {"label", "class_id", "confidence", "bbox"}
        assert d["label"] == "Central Incisor"


# ---------------------------------------------------------------------------
# DetectionResult
# ---------------------------------------------------------------------------

class TestDetectionResult:
    def test_len(self, sample_result):
        assert len(sample_result) == 1

    def test_labels(self, sample_result):
        assert sample_result.labels == ["Central Incisor"]

    def test_unique_labels(self):
        r = DetectionResult(
            detections=[
                Detection("Canine", 4, 0.8, (0, 0, 10, 10)),
                Detection("Canine", 4, 0.7, (10, 10, 20, 20)),
                Detection("1st Molar", 0, 0.9, (20, 20, 30, 30)),
            ],
        )
        assert r.unique_labels == ["1st Molar", "Canine"]

    def test_filter_by_label(self, sample_result):
        filtered = sample_result.filter_by_label("Central Incisor")
        assert len(filtered) == 1
        filtered_empty = sample_result.filter_by_label("Canine")
        assert len(filtered_empty) == 0

    def test_filter_by_confidence(self):
        r = DetectionResult(
            detections=[
                Detection("Canine", 4, 0.9, (0, 0, 10, 10)),
                Detection("1st Molar", 0, 0.4, (20, 20, 30, 30)),
            ],
        )
        high = r.filter_by_confidence(0.6)
        assert len(high) == 1
        assert high.detections[0].label == "Canine"

    def test_to_dict_structure(self, sample_result):
        d = sample_result.to_dict()
        assert "image_size" in d
        assert "count" in d
        assert "detections" in d
        assert d["count"] == 1

    def test_iteration(self, sample_result):
        items = list(sample_result)
        assert len(items) == 1
        assert items[0].label == "Central Incisor"


# ---------------------------------------------------------------------------
# TOOTH_CLASSES
# ---------------------------------------------------------------------------

def test_tooth_classes_completeness():
    assert len(TOOTH_CLASSES) == 7
    assert 0 in TOOTH_CLASSES and TOOTH_CLASSES[0] == "1st Molar"
    assert 5 in TOOTH_CLASSES and TOOTH_CLASSES[5] == "Central Incisor"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_load_from_numpy_rgb(self, dummy_image_rgb):
        bgr = load_image(dummy_image_rgb)
        assert bgr.shape == (80, 100, 3)

    def test_load_from_pil(self, dummy_pil_image):
        bgr = load_image(dummy_pil_image)
        assert bgr.ndim == 3
        assert bgr.shape[2] == 3

    def test_load_from_path(self, tmp_path, dummy_pil_image):
        p = tmp_path / "test.jpg"
        dummy_pil_image.save(p)
        bgr = load_image(p)
        assert bgr.ndim == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            load_image(12345)


class TestLetterbox:
    def test_output_shape(self, dummy_image_rgb):
        import cv2
        bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR)
        out, scale, (left, top) = letterbox(bgr, (640, 640))
        assert out.shape == (640, 640, 3)

    def test_aspect_ratio_preserved(self):
        import cv2
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        out, scale, (left, top) = letterbox(img, (640, 640))
        assert out.shape == (640, 640, 3)
        assert scale == pytest.approx(3.2)


class TestToPil:
    def test_returns_pil_image(self, dummy_image_rgb):
        import cv2
        bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR)
        pil = to_pil(bgr)
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestAnnotate:
    def test_returns_same_shape(self, dummy_image_rgb, sample_result):
        import cv2
        bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR)
        out = annotate(bgr, sample_result)
        assert out.shape == bgr.shape

    def test_does_not_mutate_input(self, dummy_image_rgb, sample_result):
        import cv2
        bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR)
        original = bgr.copy()
        annotate(bgr, sample_result)
        np.testing.assert_array_equal(bgr, original)

    def test_empty_result_unchanged(self, dummy_image_rgb):
        import cv2
        bgr = cv2.cvtColor(dummy_image_rgb, cv2.COLOR_RGB2BGR)
        empty = DetectionResult(detections=[], image_width=100, image_height=80)
        out = annotate(bgr, empty)
        np.testing.assert_array_equal(out, bgr)


# ---------------------------------------------------------------------------
# DentalDetector (unit — YOLO mocked)
# ---------------------------------------------------------------------------

def _make_mock_model(detections):
    """Build a mock ultralytics YOLO result."""
    import torch

    boxes_mock = MagicMock()
    boxes_mock.xyxy.cpu().numpy.return_value = np.array(
        [[d["bbox"]] for d in detections], dtype=np.float32
    ).reshape(-1, 4) if detections else np.empty((0, 4), dtype=np.float32)
    boxes_mock.conf.cpu().numpy.return_value = np.array(
        [d["conf"] for d in detections], dtype=np.float32
    )
    boxes_mock.cls.cpu().numpy.return_value = np.array(
        [d["cls"] for d in detections], dtype=np.float32
    )

    result_mock = MagicMock()
    result_mock.boxes = boxes_mock
    result_mock.names = {int(k): v for k, v in TOOTH_CLASSES.items()}

    model_mock = MagicMock()
    model_mock.return_value = [result_mock]
    return model_mock


@pytest.fixture
def mock_detector(tmp_path):
    """DentalDetector with a dummy weights file and mocked YOLO."""
    weights = tmp_path / "dummy.pt"
    weights.touch()

    with patch("dental_detector.detector.YOLO") as MockYOLO:
        mock_model = _make_mock_model([
            {"bbox": (5, 10, 50, 60), "conf": 0.88, "cls": 5},
            {"bbox": (60, 10, 95, 60), "conf": 0.55, "cls": 4},
        ])
        MockYOLO.return_value = mock_model
        detector = DentalDetector(str(weights), conf_threshold=0.6)
        detector._model = mock_model
        yield detector


class TestDentalDetector:
    def test_detect_returns_result(self, mock_detector, dummy_image_rgb):
        result = mock_detector.detect(dummy_image_rgb)
        assert isinstance(result, DetectionResult)

    def test_confidence_filtering(self, mock_detector, dummy_image_rgb):
        result = mock_detector.detect(dummy_image_rgb, conf_threshold=0.6)
        for det in result:
            assert det.confidence >= 0.6

    def test_detect_to_json(self, mock_detector, dummy_image_rgb):
        js = mock_detector.detect_to_json(dummy_image_rgb)
        data = json.loads(js)
        assert "detections" in data
        assert "count" in data

    def test_annotate_returns_pil(self, mock_detector, dummy_image_rgb):
        pil = mock_detector.annotate(dummy_image_rgb)
        assert isinstance(pil, Image.Image)

    def test_annotate_and_save(self, mock_detector, dummy_image_rgb, tmp_path):
        out = tmp_path / "out.jpg"
        returned = mock_detector.annotate_and_save(dummy_image_rgb, out)
        assert returned == out
        assert out.exists()

    def test_model_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            DentalDetector("/nonexistent/model.pt")

    def test_repr(self, mock_detector):
        r = repr(mock_detector)
        assert "DentalDetector" in r
        assert "conf_threshold" in r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_detect_prints_output(self, mock_detector, dummy_pil_image, tmp_path, capsys):
        img_path = tmp_path / "test.jpg"
        dummy_pil_image.save(img_path)
        weights = tmp_path / "dummy.pt"
        weights.touch()

        from dental_detector.cli import main

        with patch("dental_detector.cli.DentalDetector") as MockDet:
            inst = MagicMock()
            inst.detect.return_value = DetectionResult(
                detections=[Detection("1st Molar", 0, 0.9, (0, 0, 10, 10))],
                image_width=64,
                image_height=64,
            )
            MockDet.return_value = inst
            rc = main([str(img_path), "--model", str(weights)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "1st Molar" in out

    def test_json_flag(self, dummy_pil_image, tmp_path, capsys):
        img_path = tmp_path / "test.jpg"
        dummy_pil_image.save(img_path)
        weights = tmp_path / "dummy.pt"
        weights.touch()

        from dental_detector.cli import main

        with patch("dental_detector.cli.DentalDetector") as MockDet:
            inst = MagicMock()
            inst.detect.return_value = DetectionResult(
                detections=[Detection("Canine", 4, 0.75, (5, 5, 20, 20))],
                image_width=64,
                image_height=64,
            )
            MockDet.return_value = inst
            rc = main([str(img_path), "--model", str(weights), "--json"])

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "detections" in data
