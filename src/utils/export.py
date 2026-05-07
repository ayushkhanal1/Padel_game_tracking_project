from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "frame",
    "timestamp",
    "class",
    "model_class",
    "track_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def flatten_detection(record: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = record["bbox"]
    return {
        "frame": record["frame"],
        "timestamp": record["timestamp"],
        "class": record["class"],
        "model_class": record["model_class"],
        "track_id": record["track_id"],
        "confidence": record["confidence"],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def write_detections_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(flatten_detection(record) for record in records)


def write_detections_json(
    records: list[dict[str, Any]],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "detections": records,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
