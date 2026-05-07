from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_OUTPUT_DIR  # noqa: E402
from src.utils.export import write_json  # noqa: E402


DEFAULT_SHOTS = DEFAULT_OUTPUT_DIR / "shots_phase2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shot count analytics and a simple dashboard."
    )
    parser.add_argument(
        "--shots",
        type=Path,
        default=DEFAULT_SHOTS,
        help=f"Phase 2 shot JSON. Default: {DEFAULT_SHOTS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--analytics-name",
        default="analytics_phase3.json",
        help="Analytics JSON output filename.",
    )
    parser.add_argument(
        "--counts-name",
        default="shot_counts_phase3.csv",
        help="Shot type counts CSV output filename.",
    )
    parser.add_argument(
        "--players-name",
        default="player_summary_phase3.csv",
        help="Per-player summary CSV output filename.",
    )
    parser.add_argument(
        "--timeline-name",
        default="shot_timeline_phase3.csv",
        help="Shot timeline CSV output filename.",
    )
    parser.add_argument(
        "--dashboard-name",
        default="dashboard_phase3.png",
        help="Dashboard image output filename.",
    )
    return parser.parse_args()


def load_shots(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("shots", []), payload.get("metadata", {}), payload.get("summary", {})


def safe_player_id(shot: dict[str, Any]) -> str:
    player_id = shot.get("player_track_id")
    return "unknown" if player_id in (None, "") else str(player_id)


def build_timeline(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline = []
    previous_timestamp: float | None = None
    for shot in sorted(shots, key=lambda item: int(item["frame"])):
        timestamp = float(shot["timestamp"])
        gap = None if previous_timestamp is None else round(timestamp - previous_timestamp, 3)
        timeline.append(
            {
                "shot_id": int(shot["shot_id"]),
                "timestamp": round(timestamp, 3),
                "frame": int(shot["frame"]),
                "shot_type": shot["shot_type"],
                "player_track_id": safe_player_id(shot),
                "player_side": shot.get("player_side", "unknown"),
                "event_confidence": float(shot["event_confidence"]),
                "seconds_since_previous": gap,
            }
        )
        previous_timestamp = timestamp
    return timeline


def build_player_summary(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for shot in shots:
        grouped[safe_player_id(shot)].append(shot)

    rows = []
    for player_id, player_shots in sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        shot_type_counts = Counter(shot["shot_type"] for shot in player_shots)
        side_counts = Counter(shot.get("player_side", "unknown") for shot in player_shots)
        avg_confidence = sum(float(shot["event_confidence"]) for shot in player_shots) / len(player_shots)
        rows.append(
            {
                "player_track_id": player_id,
                "total_shots": len(player_shots),
                "forehand": shot_type_counts.get("forehand", 0),
                "backhand": shot_type_counts.get("backhand", 0),
                "serve_or_smash": shot_type_counts.get("serve_or_smash", 0),
                "unknown": shot_type_counts.get("unknown", 0),
                "near_side_shots": side_counts.get("near", 0),
                "far_side_shots": side_counts.get("far", 0),
                "avg_event_confidence": round(avg_confidence, 4),
            }
        )
    return rows


def build_analytics(
    shots: list[dict[str, Any]],
    metadata: dict[str, Any],
    phase2_summary: dict[str, Any],
) -> dict[str, Any]:
    shot_type_counts = Counter(shot["shot_type"] for shot in shots)
    side_counts = Counter(shot.get("player_side", "unknown") for shot in shots)
    player_counts = Counter(safe_player_id(shot) for shot in shots)
    timeline = build_timeline(shots)
    player_summary = build_player_summary(shots)

    duration = float(metadata.get("duration_seconds") or 0.0)
    shot_rate = (len(shots) / duration * 60.0) if duration > 0 else 0.0
    avg_confidence = (
        sum(float(shot["event_confidence"]) for shot in shots) / len(shots)
        if shots
        else 0.0
    )

    return {
        "metadata": metadata,
        "phase2_summary": phase2_summary,
        "summary": {
            "total_shots": len(shots),
            "duration_seconds": round(duration, 3),
            "shot_rate_per_minute": round(shot_rate, 3),
            "avg_event_confidence": round(avg_confidence, 4),
            "shot_type_counts": dict(shot_type_counts),
            "player_side_counts": dict(side_counts),
            "player_track_counts": dict(player_counts),
        },
        "shot_counts": [
            {"shot_type": shot_type, "count": count}
            for shot_type, count in sorted(shot_type_counts.items())
        ],
        "player_summary": player_summary,
        "timeline": timeline,
    }


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_bar_chart(
    canvas: np.ndarray,
    title: str,
    counts: dict[str, int],
    origin: tuple[int, int],
    size: tuple[int, int],
    colors: list[tuple[int, int, int]],
) -> None:
    x, y = origin
    width, height = size
    draw_text(canvas, title, (x, y - 18), 0.72, (245, 245, 245), 2)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (55, 55, 55), 1)

    if not counts:
        draw_text(canvas, "No data", (x + 24, y + 60), 0.65, (200, 200, 200), 2)
        return

    max_count = max(counts.values()) or 1
    row_height = max(34, height // max(1, len(counts)))
    for idx, (label, count) in enumerate(sorted(counts.items(), key=lambda item: (-item[1], item[0]))):
        row_y = y + idx * row_height + 16
        bar_width = int((width - 180) * count / max_count)
        color = colors[idx % len(colors)]
        draw_text(canvas, label.replace("_", " ").title(), (x + 18, row_y + 12), 0.55, (235, 235, 235), 2)
        cv2.rectangle(canvas, (x + 205, row_y - 10), (x + 205 + bar_width, row_y + 14), color, -1)
        draw_text(canvas, str(count), (x + 215 + bar_width, row_y + 12), 0.55, (245, 245, 245), 2)


def draw_timeline(
    canvas: np.ndarray,
    shots: list[dict[str, Any]],
    duration_seconds: float,
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    x, y = origin
    width, height = size
    draw_text(canvas, "Shot Timeline", (x, y - 18), 0.72, (245, 245, 245), 2)
    cv2.line(canvas, (x, y + height // 2), (x + width, y + height // 2), (120, 120, 120), 2)

    colors = {
        "forehand": (55, 180, 255),
        "backhand": (80, 220, 130),
        "serve_or_smash": (0, 220, 255),
        "unknown": (180, 180, 180),
    }
    duration = max(duration_seconds, 1.0)
    for shot in shots:
        timestamp = float(shot["timestamp"])
        shot_x = x + int(width * timestamp / duration)
        shot_type = str(shot["shot_type"])
        color = colors.get(shot_type, colors["unknown"])
        cv2.circle(canvas, (shot_x, y + height // 2), 8, color, -1)
        draw_text(canvas, str(shot["shot_id"]), (shot_x - 6, y + height // 2 - 16), 0.42, (245, 245, 245), 1)


def render_dashboard(analytics: dict[str, Any], shots: list[dict[str, Any]], path: Path) -> None:
    canvas = np.full((720, 1280, 3), (28, 29, 32), dtype=np.uint8)
    summary = analytics["summary"]
    shot_counts = summary["shot_type_counts"]
    side_counts = summary["player_side_counts"]

    draw_text(canvas, "Padel Game Analytics", (44, 64), 1.2, (245, 245, 245), 2)
    draw_text(canvas, "Prototype shot detection summary", (46, 102), 0.66, (190, 190, 190), 2)

    cards = [
        ("Total Shots", str(summary["total_shots"])),
        ("Shot Rate / Min", f"{summary['shot_rate_per_minute']:.2f}"),
        ("Avg Confidence", f"{summary['avg_event_confidence']:.2f}"),
    ]
    for idx, (label, value) in enumerate(cards):
        x = 44 + idx * 250
        y = 134
        cv2.rectangle(canvas, (x, y), (x + 220, y + 112), (42, 43, 48), -1)
        cv2.rectangle(canvas, (x, y), (x + 220, y + 112), (75, 75, 82), 1)
        draw_text(canvas, label, (x + 18, y + 34), 0.55, (190, 190, 190), 2)
        draw_text(canvas, value, (x + 18, y + 84), 1.12, (255, 255, 255), 2)

    palette = [(55, 180, 255), (80, 220, 130), (0, 220, 255), (180, 180, 180)]
    draw_bar_chart(canvas, "Shot Types", shot_counts, (44, 310), (560, 205), palette)
    draw_bar_chart(canvas, "Player Side", side_counts, (675, 310), (360, 150), palette)
    draw_timeline(canvas, shots, float(summary["duration_seconds"]), (44, 610), (1120, 65))

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def main() -> None:
    args = parse_args()
    shots, metadata, phase2_summary = load_shots(args.shots)
    analytics = build_analytics(shots, metadata, phase2_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analytics_path = args.output_dir / args.analytics_name
    counts_path = args.output_dir / args.counts_name
    players_path = args.output_dir / args.players_name
    timeline_path = args.output_dir / args.timeline_name
    dashboard_path = args.output_dir / args.dashboard_name

    write_json(analytics, analytics_path)
    write_csv(analytics["shot_counts"], counts_path, ["shot_type", "count"])
    write_csv(
        analytics["player_summary"],
        players_path,
        [
            "player_track_id",
            "total_shots",
            "forehand",
            "backhand",
            "serve_or_smash",
            "unknown",
            "near_side_shots",
            "far_side_shots",
            "avg_event_confidence",
        ],
    )
    write_csv(
        analytics["timeline"],
        timeline_path,
        [
            "shot_id",
            "timestamp",
            "frame",
            "shot_type",
            "player_track_id",
            "player_side",
            "event_confidence",
            "seconds_since_previous",
        ],
    )
    render_dashboard(analytics, shots, dashboard_path)

    print(f"Analytics JSON: {analytics_path}")
    print(f"Shot counts CSV: {counts_path}")
    print(f"Player summary CSV: {players_path}")
    print(f"Timeline CSV: {timeline_path}")
    print(f"Dashboard image: {dashboard_path}")
    print(f"Total shots: {analytics['summary']['total_shots']}")


if __name__ == "__main__":
    main()
