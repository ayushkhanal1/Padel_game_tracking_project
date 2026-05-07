from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_OUTPUT_DIR, DEFAULT_SOURCE  # noqa: E402
from src.utils.export import write_json  # noqa: E402
from src.utils.video import load_video_info, make_video_writer  # noqa: E402


DEFAULT_DETECTIONS = DEFAULT_OUTPUT_DIR / "detections_phase1.json"
DEFAULT_PHASE1_VIDEO = DEFAULT_OUTPUT_DIR / "annotated_phase1.mp4"


@dataclass(frozen=True)
class Detection:
    frame: int
    timestamp: float
    class_name: str
    model_class: str
    track_id: int | None
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


SHOT_FIELDS = [
    "shot_id",
    "frame",
    "timestamp",
    "shot_type",
    "player_track_id",
    "player_side",
    "ball_track_id",
    "ball_confidence",
    "event_confidence",
    "nearest_racket_track_id",
    "racket_distance",
    "player_distance",
    "ball_x",
    "ball_y",
    "classification_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract rule-based padel shot events from Phase 1 detections."
    )
    parser.add_argument(
        "--detections",
        type=Path,
        default=DEFAULT_DETECTIONS,
        help=f"Phase 1 detection JSON. Default: {DEFAULT_DETECTIONS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--event-gap",
        type=int,
        default=8,
        help="Maximum frame gap for grouping dynamic ball detections into one shot window.",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=2,
        help="Frames before/after the ball frame to search for nearby rackets/players.",
    )
    parser.add_argument(
        "--max-player-distance",
        type=float,
        default=190.0,
        help="Maximum ball-to-player center distance for accepting a shot event.",
    )
    parser.add_argument(
        "--max-racket-distance",
        type=float,
        default=150.0,
        help="Maximum ball-to-racket center distance for accepting a shot event.",
    )
    parser.add_argument(
        "--csv-name",
        default="shots_phase2.csv",
        help="Shot CSV output filename.",
    )
    parser.add_argument(
        "--json-name",
        default="shots_phase2.json",
        help="Shot JSON output filename.",
    )
    parser.add_argument(
        "--video-source",
        type=Path,
        default=DEFAULT_PHASE1_VIDEO,
        help=(
            "Video to annotate with shot labels. Defaults to the Phase 1 "
            f"annotated video: {DEFAULT_PHASE1_VIDEO}"
        ),
    )
    parser.add_argument(
        "--video-name",
        default="annotated_phase2.mp4",
        help="Phase 2 annotated output video filename.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the Phase 2 annotated video.",
    )
    return parser.parse_args()


def detection_from_record(record: dict[str, Any]) -> Detection:
    return Detection(
        frame=int(record["frame"]),
        timestamp=float(record["timestamp"]),
        class_name=str(record["class"]),
        model_class=str(record.get("model_class", record["class"])),
        track_id=None if record.get("track_id") is None else int(record["track_id"]),
        confidence=float(record["confidence"]),
        bbox=tuple(float(value) for value in record["bbox"]),
    )


def load_detections(path: Path) -> tuple[list[Detection], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    detections = [detection_from_record(record) for record in payload["detections"]]
    return detections, payload.get("metadata", {})


def distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def distance_to_point(detection: Detection, x: float, y: float) -> float:
    return math.hypot(detection.cx - x, detection.cy - y)


def group_by_frame(detections: list[Detection]) -> dict[int, list[Detection]]:
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_frame[detection.frame].append(detection)
    return by_frame


def ball_track_stats(ball_detections: list[Detection]) -> dict[int, dict[str, float]]:
    by_track: dict[int, list[Detection]] = defaultdict(list)
    for detection in ball_detections:
        if detection.track_id is not None:
            by_track[detection.track_id].append(detection)

    stats: dict[int, dict[str, float]] = {}
    for track_id, track in by_track.items():
        ordered = sorted(track, key=lambda detection: detection.frame)
        xs = [detection.cx for detection in ordered]
        ys = [detection.cy for detection in ordered]
        speeds = []
        for prev, current in zip(ordered, ordered[1:]):
            frame_delta = max(1, current.frame - prev.frame)
            speeds.append(distance(prev, current) / frame_delta)

        stats[track_id] = {
            "frames": len({detection.frame for detection in ordered}),
            "range_x": max(xs) - min(xs),
            "range_y": max(ys) - min(ys),
            "displacement": math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]),
            "max_speed": max(speeds) if speeds else 0.0,
            "mean_confidence": sum(detection.confidence for detection in ordered) / len(ordered),
        }
    return stats


def static_ball_track_ids(stats: dict[int, dict[str, float]]) -> set[int]:
    static_ids = set()
    for track_id, values in stats.items():
        if (
            values["frames"] >= 10
            and values["displacement"] < 12
            and values["max_speed"] < 2.0
        ):
            static_ids.add(track_id)
    return static_ids


def cluster_frames(frames: list[int], max_gap: int) -> list[list[int]]:
    clusters: list[list[int]] = []
    current: list[int] = []
    previous = None
    for frame in sorted(frames):
        if previous is None or frame - previous <= max_gap:
            current.append(frame)
        else:
            clusters.append(current)
            current = [frame]
        previous = frame

    if current:
        clusters.append(current)
    return clusters


def nearby_detections(
    by_frame: dict[int, list[Detection]],
    frame: int,
    class_name: str,
    window: int,
) -> list[Detection]:
    matches = []
    for candidate_frame in range(frame - window, frame + window + 1):
        for detection in by_frame.get(candidate_frame, []):
            if detection.class_name == class_name:
                matches.append(detection)
    return matches


def nearest_detection(
    detections: list[Detection],
    x: float,
    y: float,
) -> tuple[Detection | None, float | None]:
    if not detections:
        return None, None

    nearest = min(detections, key=lambda detection: distance_to_point(detection, x, y))
    return nearest, distance_to_point(nearest, x, y)


def score_candidate(
    ball: Detection,
    racket_distance: float | None,
    player_distance: float | None,
    max_racket_distance: float,
    max_player_distance: float,
) -> float:
    score = ball.confidence
    if racket_distance is not None:
        score += max(0.0, 1.0 - racket_distance / max_racket_distance) * 0.8
    if player_distance is not None:
        score += max(0.0, 1.0 - player_distance / max_player_distance) * 0.5
    return score


def player_side(player: Detection | None, video_height: float) -> str:
    if player is None or video_height <= 0:
        return "unknown"
    return "near" if player.cy >= video_height * 0.45 else "far"


def classify_shot(
    *,
    shot_index: int,
    frame: int,
    fps: float,
    player: Detection | None,
    ball: Detection,
    side: str,
) -> tuple[str, str]:
    if shot_index == 0 and frame <= fps * 4:
        return "serve_or_smash", "first accepted shot event occurs in the opening seconds"

    if player is None:
        return "unknown", "no reliable player assignment"

    relative_x = ball.cx - player.cx
    contact_high = ball.cy < player.bbox[1] + player.height * 0.35
    if contact_high:
        return "serve_or_smash", "ball is high relative to the assigned player"

    if side == "far":
        is_forehand = relative_x < 0
    else:
        is_forehand = relative_x > 0

    if is_forehand:
        return "forehand", "ball is on the estimated forehand side of the player"
    return "backhand", "ball is on the estimated backhand side of the player"


def event_confidence(
    ball: Detection,
    racket_distance: float | None,
    player_distance: float | None,
    max_racket_distance: float,
    max_player_distance: float,
) -> float:
    score = ball.confidence * 0.45
    if racket_distance is not None:
        score += max(0.0, 1.0 - racket_distance / max_racket_distance) * 0.35
    if player_distance is not None:
        score += max(0.0, 1.0 - player_distance / max_player_distance) * 0.20
    return round(min(1.0, score), 4)


def choose_cluster_event(
    cluster_frames_: list[int],
    ball_candidates: list[Detection],
    by_frame: dict[int, list[Detection]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    cluster_set = set(cluster_frames_)
    candidates = [ball for ball in ball_candidates if ball.frame in cluster_set]
    scored = []

    for ball in candidates:
        rackets = nearby_detections(by_frame, ball.frame, "racket", args.neighbor_window)
        players = nearby_detections(by_frame, ball.frame, "person", args.neighbor_window)

        racket, racket_distance = nearest_detection(rackets, ball.cx, ball.cy)
        if racket_distance is not None and racket_distance > args.max_racket_distance:
            racket = None
            racket_distance = None

        player_anchor_x = racket.cx if racket is not None else ball.cx
        player_anchor_y = racket.cy if racket is not None else ball.cy
        player, player_distance = nearest_detection(players, player_anchor_x, player_anchor_y)
        if player_distance is not None and player_distance > args.max_player_distance:
            player = None
            player_distance = None

        if racket is None and player is None:
            continue

        scored.append(
            (
                score_candidate(
                    ball,
                    racket_distance,
                    player_distance,
                    args.max_racket_distance,
                    args.max_player_distance,
                ),
                ball,
                racket,
                racket_distance,
                player,
                player_distance,
            )
        )

    if not scored:
        return None

    _, ball, racket, racket_distance, player, player_distance = max(
        scored,
        key=lambda item: item[0],
    )
    return {
        "ball": ball,
        "racket": racket,
        "racket_distance": racket_distance,
        "player": player,
        "player_distance": player_distance,
    }


def extract_shots(
    detections: list[Detection],
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_frame = group_by_frame(detections)
    ball_detections = [detection for detection in detections if detection.class_name == "ball"]
    stats = ball_track_stats(ball_detections)
    static_ids = static_ball_track_ids(stats)
    dynamic_balls = [
        detection
        for detection in ball_detections
        if detection.track_id is None or detection.track_id not in static_ids
    ]

    frames = sorted({detection.frame for detection in dynamic_balls})
    clusters = cluster_frames(frames, args.event_gap)

    fps = float(metadata.get("fps") or 30.0)
    video_height = float(metadata.get("height") or 0.0)
    shots: list[dict[str, Any]] = []

    for cluster in clusters:
        event = choose_cluster_event(cluster, dynamic_balls, by_frame, args)
        if event is None:
            continue

        ball = event["ball"]
        player = event["player"]
        racket = event["racket"]
        side = player_side(player, video_height)
        shot_type, reason = classify_shot(
            shot_index=len(shots),
            frame=ball.frame,
            fps=fps,
            player=player,
            ball=ball,
            side=side,
        )

        shots.append(
            {
                "shot_id": len(shots) + 1,
                "frame": ball.frame,
                "timestamp": round(ball.frame / fps, 3),
                "shot_type": shot_type,
                "player_track_id": None if player is None else player.track_id,
                "player_side": side,
                "ball_track_id": ball.track_id,
                "ball_confidence": round(ball.confidence, 4),
                "event_confidence": event_confidence(
                    ball,
                    event["racket_distance"],
                    event["player_distance"],
                    args.max_racket_distance,
                    args.max_player_distance,
                ),
                "nearest_racket_track_id": None if racket is None else racket.track_id,
                "racket_distance": None
                if event["racket_distance"] is None
                else round(event["racket_distance"], 2),
                "player_distance": None
                if event["player_distance"] is None
                else round(event["player_distance"], 2),
                "ball_x": round(ball.cx, 2),
                "ball_y": round(ball.cy, 2),
                "ball_bbox": [round(value, 2) for value in ball.bbox],
                "classification_reason": reason,
            }
        )

    summary = {
        "shot_count": len(shots),
        "shot_type_counts": dict(Counter(shot["shot_type"] for shot in shots)),
        "static_ball_track_ids_removed": sorted(static_ids),
        "dynamic_ball_frames": len(frames),
        "ball_track_stats": {
            str(track_id): {
                key: round(value, 4) for key, value in values.items()
            }
            for track_id, values in sorted(stats.items())
        },
        "parameters": {
            "event_gap": args.event_gap,
            "neighbor_window": args.neighbor_window,
            "max_player_distance": args.max_player_distance,
            "max_racket_distance": args.max_racket_distance,
        },
    }
    return shots, summary


def write_shots_csv(shots: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHOT_FIELDS)
        writer.writeheader()
        for shot in shots:
            writer.writerow({field: shot.get(field) for field in SHOT_FIELDS})


def readable_shot_type(shot_type: str) -> str:
    return shot_type.replace("_", " ").title()


def draw_label_box(
    frame: Any,
    origin: tuple[int, int],
    title: str,
    subtitle: str,
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    width = 520
    height = 92
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
    cv2.putText(
        frame,
        title,
        (x + 18, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (x + 18, y + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )


def draw_marker(
    frame: Any,
    x: float,
    y: float,
    color: tuple[int, int, int],
) -> None:
    center = (int(round(x)), int(round(y)))
    cv2.circle(frame, center, 18, color, 3)
    cv2.circle(frame, center, 4, color, -1)


def render_shot_video(
    *,
    video_source: Path,
    output_path: Path,
    shots: list[dict[str, Any]],
    display_frames_before: int = 12,
    display_frames_after: int = 42,
) -> None:
    if not video_source.exists() and DEFAULT_SOURCE.exists():
        video_source = DEFAULT_SOURCE

    video_info = load_video_info(video_source)
    capture = cv2.VideoCapture(str(video_source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for shot overlay: {video_source}")

    writer = make_video_writer(
        output_path,
        video_info.fps,
        video_info.width,
        video_info.height,
    )
    # Intentionally no shot classification overlay in the video. Shot labels are
    # provided via the structured output files.

    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()


def main() -> None:
    args = parse_args()
    detections, metadata = load_detections(args.detections)
    shots, summary = extract_shots(detections, metadata, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / args.csv_name
    json_path = args.output_dir / args.json_name
    video_path = args.output_dir / args.video_name

    write_shots_csv(shots, csv_path)
    write_json(
        {
            "metadata": metadata,
            "summary": summary,
            "shots": shots,
        },
        json_path,
    )
    if not args.no_video:
        render_shot_video(
            video_source=args.video_source,
            output_path=video_path,
            shots=shots,
        )

    print(f"Shot CSV: {csv_path}")
    print(f"Shot JSON: {json_path}")
    if not args.no_video:
        print(f"Annotated Phase 2 video: {video_path}")
    print(f"Shots detected: {len(shots)}")
    for shot_type, count in summary["shot_type_counts"].items():
        print(f"  {shot_type}: {count}")
    if summary["static_ball_track_ids_removed"]:
        removed = ", ".join(str(track_id) for track_id in summary["static_ball_track_ids_removed"])
        print(f"Removed static ball track IDs: {removed}")


if __name__ == "__main__":
    main()
