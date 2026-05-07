# Padel Game Tracking Prototype
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




## Approach

The project uses a practical computer vision pipeline instead of a large custom
training workflow. The goal is to produce a working, explainable prototype that
can be inspected and improved.

1. Detection And Tracking

`src/detect_track.py` runs Ultralytics YOLO with ByteTrack. The model detections
are mapped into the project target classes:

- `person` -> player
- `sports ball` -> ball
- `tennis racket` -> racket

Each tracked object is exported with frame number, timestamp, class, confidence,
track ID, and bounding box. This creates both structured detection files and an
annotated video.

2. Shot Event Extraction

`src/analyze_shots.py` reads the tracked detections and focuses on ball motion.
It filters noisy/static ball tracks, groups dynamic ball detections into event
windows, and selects representative frames for likely shots.

For each event, the code finds the nearest racket and nearest player around the
event frame. This gives every shot a timestamp, frame number, likely player
track ID, player side, ball track ID, and nearest racket track ID.

3. Shot Classification

Shot labels are assigned with lightweight rule-based logic. The classifier uses
relative ball, racket, and player geometry:

- overhead or high-motion events are labeled `serve_or_smash`
- remaining events are split into `forehand` or `backhand`

This approach is simple, transparent, and works without a labeled padel-specific
training dataset.

4. Analytics And Packaging

`src/summarize_analytics.py` creates shot-count summaries, player-side counts,
timeline data, and a dashboard image. The final packaging step then writes the
project outputs into `data/final/` with consistent names.




## Methodology
The pipeline starts with Ultralytics YOLO for object detection and ByteTrack for
track ID assignment. COCO class labels are mapped into the assignment targets:
`person` becomes player, `sports ball` becomes ball, and `tennis racket` becomes
racket. Each retained detection is exported with frame number, timestamp, class,
track ID, confidence, and bounding box.

Shot extraction is rule-based. The analyzer filters static or noisy ball tracks,
groups dynamic ball observations into candidate shot windows, and selects an
event frame for each window. Around that event frame, it finds the nearest racket
and nearest player track. This gives each shot a timestamp, frame, likely player
track ID, and side of court.

Shot classification uses lightweight geometry heuristics. Events with stronger
vertical motion or overhead-like geometry are labeled `serve_or_smash`. Remaining
events are split between `forehand` and `backhand` using the relative position of
the ball, racket, and player. This keeps the model explainable and avoids needing
a large labeled padel-specific dataset.

Finally, the pipeline exports predictions to CSV and JSON, creates shot-count
analytics, renders a dashboard image, and packages the annotated demo video into
`data/final/`.

-Challenges
Padel footage can make small-object detection difficult. The ball and racket are
small, fast, and often blurred, while wide camera angles create occlusion and
many background objects. A pretrained general-purpose YOLO model can provide a
useful baseline, but it is not specialized for padel equipment or court context.

Track IDs can also switch when players overlap, move quickly, or leave the frame.
Because the current prototype uses nearest-neighbor geometry for player
assignment, a missed racket or ball detection can affect the final shot label.

-Improvements
The most useful next improvement would be a small custom labeled dataset for
padel ball and racket detection. Fine-tuning YOLO on that data would reduce
missed detections and false positives. A second improvement would be temporal
classification using short clips around each shot event, which could learn
motion patterns for forehands, backhands, serves, and smashes more robustly than
handwritten geometry rules.

Additional improvements would include bounce detection, court-line calibration,
player identity smoothing, and confidence-based review flags for uncertain shot
events.







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
