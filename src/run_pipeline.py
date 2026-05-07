from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE,
    DEFAULT_TRACKER,
    PROJECT_ROOT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full padel analytics prototype pipeline."
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
        help=f"Temporary work directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "final",
        help="Submission-ready final output directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO model name or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--tracker",
        default=str(DEFAULT_TRACKER),
        help=f"Ultralytics tracker config. Default: {DEFAULT_TRACKER}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"YOLO confidence threshold. Default: {DEFAULT_CONFIDENCE}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"YOLO image size. Default: {DEFAULT_IMAGE_SIZE}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional inference device, for example 'cpu', '0', or 'cuda:0'.",
    )
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Reuse existing work detections instead of rerunning YOLO.",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep temporary work files after creating data/final.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("\nRunning:", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def can_clean_work_dir(path: Path) -> bool:
    resolved = path.resolve()
    data_dir = (PROJECT_ROOT / "data").resolve()
    protected = {
        (PROJECT_ROOT / "data" / "raw").resolve(),
        (PROJECT_ROOT / "data" / "final").resolve(),
    }
    return resolved != data_dir and resolved not in protected and data_dir in resolved.parents


def clean_work_dir(path: Path) -> None:
    if not path.exists():
        return
    if not can_clean_work_dir(path):
        raise ValueError(f"Refusing to clean unexpected work directory: {path}")
    shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detections_path = args.output_dir / "detections_phase1.json"
    phase1_video = args.output_dir / "annotated_phase1.mp4"
    shots_path = args.output_dir / "shots_phase2.json"

    if not args.skip_detect:
        detect_command = [
            sys.executable,
            "-m",
            "src.detect_track",
            "--source",
            str(args.source),
            "--output-dir",
            str(args.output_dir),
            "--model",
            str(args.model),
            "--tracker",
            str(args.tracker),
            "--conf",
            str(args.conf),
            "--imgsz",
            str(args.imgsz),
        ]
        if args.device:
            detect_command.extend(["--device", args.device])
        run_command(detect_command)
    elif not detections_path.exists():
        raise FileNotFoundError(
            f"--skip-detect was used but detections are missing: {detections_path}"
        )

    phase2_command = [
        sys.executable,
        "-m",
        "src.analyze_shots",
        "--detections",
        str(detections_path),
        "--output-dir",
        str(args.output_dir),
        "--video-source",
        str(phase1_video if phase1_video.exists() else args.source),
    ]
    run_command(phase2_command)

    analytics_command = [
        sys.executable,
        "-m",
        "src.summarize_analytics",
        "--shots",
        str(shots_path),
        "--output-dir",
        str(args.output_dir),
    ]
    run_command(analytics_command)

    final_command = [
        sys.executable,
        "-m",
        "src.finalize_submission",
        "--outputs-dir",
        str(args.output_dir),
        "--final-dir",
        str(args.final_dir),
    ]
    run_command(final_command)

    if not args.keep_work:
        clean_work_dir(args.output_dir)

    print("\nPipeline complete. Main outputs:")
    print(f"  Final folder: {args.final_dir}")
    print(f"  Demo video: {args.final_dir / 'demo_video.mp4'}")
    print(f"  Shot CSV: {args.final_dir / 'shot_predictions.csv'}")
    print(f"  Shot JSON: {args.final_dir / 'shot_predictions.json'}")
    print(f"  Dashboard: {args.final_dir / 'dashboard.png'}")


if __name__ == "__main__":
    main()
