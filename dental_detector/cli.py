"""Command-line interface for dental_detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dental_detector import DentalDetector


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dental-detector",
        description="Detect teeth in dental images (maxilla / mandible) using YOLOv8.",
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument("--model", "-m", required=True, help="Path to YOLOv8 .pt weights file.")
    p.add_argument(
        "--conf", "-c", type=float, default=0.6,
        help="Confidence threshold (default: 0.6).",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Save annotated image to this path. Skipped if not provided.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Print detections as JSON instead of plain text.",
    )
    p.add_argument(
        "--no-confidence", action="store_true",
        help="Hide confidence scores in the annotated image.",
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        detector = DentalDetector(args.model, conf_threshold=args.conf)
        result = detector.detect(args.image)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Detected {len(result)} tooth/teeth in '{Path(args.image).name}':")
        for det in result:
            x1, y1, x2, y2 = det.bbox
            print(
                f"  [{det.class_id}] {det.label:<18} conf={det.confidence:.3f} "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )

    if args.output:
        out_path = detector.annotate_and_save(
            args.image,
            args.output,
            conf_threshold=args.conf,
            show_confidence=not args.no_confidence,
        )
        print(f"Annotated image saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
