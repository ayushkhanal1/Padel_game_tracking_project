# Padel Game Analytics Prototype

This project analyzes padel gameplay footage and produces shot-level analytics.
It detects and tracks players, rackets, and the ball, identifies likely shot
events, classifies each shot, and exports the results as structured CSV/JSON
files with an annotated demo video.

## Project Requirements Covered

- Detect and track the main gameplay objects: players, racket, and ball.
- Identify shot events from tracked ball and racket movement.
- Classify each shot as `forehand`, `backhand`, or `serve_or_smash`.
- Export shot predictions with `shot_type`, `timestamp`, `frame`, and
  `player_track_id`.
- Generate a demo video showing the detection and tracking output.
- Produce basic analytics such as total shots, shot-type counts, player-side
  counts, and a dashboard image.

## What Was Built

The repository contains a complete Python pipeline that runs from one command:

```bash
python -m src.run_pipeline
```

The pipeline takes an input video from `data/raw/input_video.mp4`, runs object
detection and tracking, extracts shot events, classifies them with simple
geometry-based rules, creates analytics, and packages the final outputs into
`data/final/`.

The current generated result contains 11 shot predictions across three shot
classes:

- `serve_or_smash`: 5
- `forehand`: 4
- `backhand`: 2

## Approach

The project uses a practical computer vision pipeline instead of a large custom
training workflow. The goal is to produce a working, explainable prototype that
can be inspected and improved.

### 1. Detection And Tracking

`src/detect_track.py` runs Ultralytics YOLO with ByteTrack. The model detections
are mapped into the project target classes:

- `person` -> player
- `sports ball` -> ball
- `tennis racket` -> racket

Each tracked object is exported with frame number, timestamp, class, confidence,
track ID, and bounding box. This creates both structured detection files and an
annotated video.

### 2. Shot Event Extraction

`src/analyze_shots.py` reads the tracked detections and focuses on ball motion.
It filters noisy/static ball tracks, groups dynamic ball detections into event
windows, and selects representative frames for likely shots.

For each event, the code finds the nearest racket and nearest player around the
event frame. This gives every shot a timestamp, frame number, likely player
track ID, player side, ball track ID, and nearest racket track ID.

### 3. Shot Classification

Shot labels are assigned with lightweight rule-based logic. The classifier uses
relative ball, racket, and player geometry:

- overhead or high-motion events are labeled `serve_or_smash`
- remaining events are split into `forehand` or `backhand`

This approach is simple, transparent, and works without a labeled padel-specific
training dataset.

### 4. Analytics And Packaging

`src/summarize_analytics.py` creates shot-count summaries, player-side counts,
timeline data, and a dashboard image. The final packaging step then writes the
project outputs into `data/final/` with consistent names.

## Quickstart

### 1. Install

```bash
python -m pip install -r requirements.txt
```

### 2. Add Input Video

Place the input video at:

```text
data/raw/input_video.mp4
```

### 3. Run

```bash
python -m src.run_pipeline
```

Useful flags:

```bash
python -m src.run_pipeline --device cpu
python -m src.run_pipeline --keep-work
python -m src.run_pipeline --tracker trackers/bytetrack_padel.yaml
```

## Final Outputs

The final project outputs are written to `data/final/`:

- `demo_video.mp4` - annotated detection and tracking video.
- `shot_predictions.csv` - compact shot predictions table.
- `shot_predictions.json` - structured shot predictions and summary metadata.
- `detections.csv` - tracked object detections.
- `detections.json` - tracked object detections with metadata.
- `analytics.json` - shot-count and per-player analytics.
- `dashboard.png` - visual analytics dashboard.
- `OUTPUT_README.md` - manifest for the final output folder.

The `shot_predictions.csv` fields are:

```text
shot_id, shot_type, timestamp, frame, player_track_id, player_side,
event_confidence, ball_track_id, nearest_racket_track_id
```

## Model Weights

The default model is `yolo11s.pt`, configured in `src/config.py`. Model weights
are binary artifacts and are ignored by Git. Keep the weight file locally at the
project root, or pass a custom model path:

```bash
python -m src.run_pipeline --model path/to/custom_weights.pt
```

Model handoff notes can be recorded in [models/README.md](models/README.md).

## Repo Structure

- `src/` - pipeline code.
- `docs/` - detailed approach explanation.
- `models/` - model notes.
- `trackers/` - optional ByteTrack configuration.
- `data/raw/` - expected input video location.
- `data/final/` - packaged project outputs.

## Limitations

This is a practical prototype, not a production sports analytics system. Ball
and racket detection are difficult in wide-angle CCTV-style footage, so output
quality depends on video resolution, camera angle, motion blur, and occlusion.

The most useful future improvement would be fine-tuning a detector on
padel-specific ball and racket examples, followed by a temporal shot classifier
trained on labeled shot clips.
