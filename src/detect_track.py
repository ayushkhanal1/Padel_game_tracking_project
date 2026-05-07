from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DEFAULT_TRACKER,
    TARGET_CLASS_ALIASES,
)
from src.utils.export import write_detections_csv, write_detections_json, write_json  # noqa: E402
from src.utils.video import load_video_info, make_video_writer  # noqa: E402


def normalize_label(label: str) -> str:
    return " ".join(label.lower().replace("_", " ").split())


def canonical_label(model_label: str) -> str:
    normalized = normalize_label(model_label)
    for canonical, aliases in TARGET_CLASS_ALIASES.items():
        if normalized in {normalize_label(alias) for alias in aliases}:
            return canonical
    return normalized


def resolve_target_class_ids(model_names: dict[int, str]) -> list[int]:
    alias_lookup = {
        normalize_label(alias)
        for aliases in TARGET_CLASS_ALIASES.values()
        for alias in aliases
    }
    class_ids = [
        int(class_id)
        for class_id, name in model_names.items()
        if normalize_label(name) in alias_lookup
    ]

    if not class_ids:
        available = ", ".join(model_names.values())
        raise ValueError(
            "No target classes were found in this model. "
            f"Expected aliases: {sorted(alias_lookup)}. Available classes: {available}"
        )

    return sorted(class_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track padel players, racket, and ball with YOLO."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input video path. Default: {DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for annotated video and detection files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ultralytics model name or local .pt path. "
            f"Default: {DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_TRACKER,
        help=f"Ultralytics tracker config. Default: {DEFAULT_TRACKER}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Detection confidence threshold. Default: {DEFAULT_CONFIDENCE}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Inference image size. Default: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device, for example 'cpu', '0', or 'cuda:0'.",
    )
    parser.add_argument(
        "--video-name",
        default="annotated_phase1.mp4",
        help="Annotated output video filename.",
    )
    parser.add_argument(
        "--csv-name",
        default="detections_phase1.csv",
        help="CSV output filename.",
    )
    parser.add_argument(
        "--json-name",
        default="detections_phase1.json",
        help="JSON output filename.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary_phase1.json",
        help="Detection summary output filename.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the annotated MP4 and only export detections.",
    )
    return parser.parse_args()


def box_to_record(
    *,
    box: Any,
    frame_index: int,
    fps: float,
    model_names: dict[int, str],
) -> dict[str, Any]:
    class_id = int(box.cls[0].item())
    model_class = model_names[class_id]
    track_id = None
    if box.id is not None:
        track_id = int(box.id[0].item())

    x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
    confidence = round(float(box.conf[0].item()), 4)

    return {
        "frame": frame_index,
        "timestamp": round(frame_index / fps, 3),
        "class": canonical_label(model_class),
        "model_class": model_class,
        "track_id": track_id,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2],
    }


def summarize_detections(
    records: list[dict[str, Any]],
    frame_count: int,
) -> dict[str, Any]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_class[record["class"]].append(record)

    classes: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for class_name in sorted(by_class):
        class_records = by_class[class_name]
        frames = {int(record["frame"]) for record in class_records}
        track_ids = {
            int(record["track_id"])
            for record in class_records
            if record["track_id"] is not None
        }
        avg_confidence = sum(record["confidence"] for record in class_records) / len(class_records)
        coverage = len(frames) / frame_count if frame_count else 0.0
        classes[class_name] = {
            "detections": len(class_records),
            "frames_with_detection": len(frames),
            "frame_coverage": round(coverage, 4),
            "unique_track_ids": len(track_ids),
            "avg_confidence": round(avg_confidence, 4),
        }

    for required_class in ("person", "racket", "ball"):
        if required_class not in classes:
            warnings.append(f"No {required_class} detections found.")
            continue

        coverage = classes[required_class]["frame_coverage"]
        if required_class in {"racket", "ball"} and coverage < 0.5:
            warnings.append(
                f"{required_class} coverage is low ({coverage:.0%}). "
                "Use the sensitive default settings or train a custom YOLO model."
            )

    return {
        "total_detections": len(records),
        "frame_count": frame_count,
        "classes": classes,
        "warnings": warnings,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\nDetection summary:")
    for class_name, stats in summary["classes"].items():
        print(
            f"  {class_name}: {stats['detections']} detections, "
            f"{stats['frames_with_detection']}/{summary['frame_count']} frames, "
            f"{stats['unique_track_ids']} track IDs, "
            f"avg conf {stats['avg_confidence']:.3f}"
        )

    if summary["warnings"]:
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


def run_tracking(args: argparse.Namespace) -> None:
    video_info = load_video_info(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = args.output_dir / args.video_name
    csv_path = args.output_dir / args.csv_name
    json_path = args.output_dir / args.json_name
    summary_path = args.output_dir / args.summary_name

    model = YOLO(args.model)
    model_names = {int(class_id): name for class_id, name in model.names.items()}
    target_class_ids = resolve_target_class_ids(model_names)

    writer = None
    if not args.no_video:
        writer = make_video_writer(
            annotated_path,
            video_info.fps,
            video_info.width,
            video_info.height,
        )

    detections: list[dict[str, Any]] = []
    results = model.track(
        source=str(args.source),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        imgsz=args.imgsz,
        classes=target_class_ids,
        device=args.device,
        verbose=False,
    )

    try:
        for frame_index, result in enumerate(results):
            if writer is not None:
                annotated_frame = result.plot()
                if annotated_frame.shape[1] != video_info.width or annotated_frame.shape[0] != video_info.height:
                    annotated_frame = cv2.resize(
                        annotated_frame,
                        (video_info.width, video_info.height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                writer.write(annotated_frame)

            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    detections.append(
                        box_to_record(
                            box=box,
                            frame_index=frame_index,
                            fps=video_info.fps,
                            model_names=model_names,
                        )
                    )

            if frame_index and frame_index % 100 == 0:
                print(f"Processed {frame_index} frames...")
    finally:
        if writer is not None:
            writer.release()

    metadata = {
        "source": str(args.source),
        "model": args.model,
        "tracker": str(args.tracker),
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "frame_count": video_info.frame_count,
        "duration_seconds": round(video_info.duration_seconds, 3),
        "target_classes": sorted(
            {
                canonical_label(model_names[class_id])
                for class_id in target_class_ids
            }
        ),
        "detections_count": len(detections),
    }

    write_detections_csv(detections, csv_path)
    write_detections_json(detections, json_path, metadata)
    summary = summarize_detections(detections, video_info.frame_count)
    summary["metadata"] = metadata
    write_json(summary, summary_path)

    print(f"CSV detections: {csv_path}")
    print(f"JSON detections: {json_path}")
    print(f"Summary: {summary_path}")
    if writer is not None:
        print(f"Annotated video: {annotated_path}")
    print(f"Detections exported: {len(detections)}")
    print_summary(summary)


def main() -> None:
    run_tracking(parse_args())


if __name__ == "__main__":
    main()
