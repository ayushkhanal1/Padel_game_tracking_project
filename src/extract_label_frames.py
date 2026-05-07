from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_SOURCE  # noqa: E402
from src.utils.video import load_video_info  # noqa: E402


DEFAULT_LABELING_DIR = Path(__file__).resolve().parents[1] / "data" / "labeling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract representative video frames for custom YOLO labeling."
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
        default=DEFAULT_LABELING_DIR,
        help=f"Labeling dataset directory. Default: {DEFAULT_LABELING_DIR}",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=20,
        help="Extract one frame every N frames.",
    )
    parser.add_argument(
        "--detections-json",
        type=Path,
        default=None,
        help="Optional detection JSON used to add nearby ball/racket frames.",
    )
    parser.add_argument(
        "--focus-classes",
        nargs="+",
        default=["ball", "racket"],
        help="Classes to oversample from detections JSON.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Frames before/after each focus detection to also extract.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Maximum number of frames to save.",
    )
    return parser.parse_args()


def load_focus_frames(
    detections_json: Path | None,
    focus_classes: set[str],
    window: int,
    frame_count: int,
) -> set[int]:
    if detections_json is None or not detections_json.exists():
        return set()

    payload = json.loads(detections_json.read_text(encoding="utf-8"))
    frames: set[int] = set()
    for record in payload.get("detections", []):
        if record.get("class") not in focus_classes:
            continue

        frame = int(record["frame"])
        for offset in range(-window, window + 1):
            candidate = frame + offset
            if 0 <= candidate < frame_count:
                frames.add(candidate)
    return frames


def limit_evenly(indices: list[int], max_frames: int) -> list[int]:
    if max_frames <= 0 or len(indices) <= max_frames:
        return indices

    step = len(indices) / max_frames
    return [indices[int(i * step)] for i in range(max_frames)]


def write_data_yaml(output_dir: Path) -> None:
    payload = "\n".join(
        [
            f"path: {output_dir.as_posix()}",
            "train: images",
            "val: images",
            "names:",
            "  0: person",
            "  1: ball",
            "  2: racket",
            "",
        ]
    )
    (output_dir / "data.yaml").write_text(payload, encoding="utf-8")


def extract_frames(args: argparse.Namespace) -> None:
    video_info = load_video_info(args.source)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    base_frames = set(range(0, video_info.frame_count, max(1, args.every_n)))
    focus_frames = load_focus_frames(
        args.detections_json,
        set(args.focus_classes),
        args.window,
        video_info.frame_count,
    )
    frame_indices = sorted(base_frames | focus_frames)
    frame_indices = limit_evenly(frame_indices, args.max_frames)

    capture = cv2.VideoCapture(str(args.source))
    saved = 0
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue

        output_path = image_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(output_path), frame)
        saved += 1
    capture.release()

    write_data_yaml(args.output_dir)
    print(f"Saved {saved} frames to {image_dir}")
    print(f"YOLO dataset config: {args.output_dir / 'data.yaml'}")


def main() -> None:
    extract_frames(parse_args())


if __name__ == "__main__":
    main()
