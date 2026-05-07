from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "input_video.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "_work"
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_TRACKER = "bytetrack.yaml"

DEFAULT_CONFIDENCE = 0.03
DEFAULT_IMAGE_SIZE = 1536

# COCO gives useful baseline classes for this phase. A custom model can still
# use direct labels such as "ball" and "racket".
TARGET_CLASS_ALIASES = {
    "person": ("person", "player"),
    "ball": ("ball", "sports ball", "tennis ball", "padel ball"),
    "racket": ("racket", "tennis racket", "padel racket"),
}
