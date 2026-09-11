"""Shared constants for the mudra recognition project."""

from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw_videos"
LANDMARKS_DIR = PROJECT_ROOT / "data" / "landmarks"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Mudra labels --------------------------------------------------------
STATIC_MUDRAS = ["pataka", "tripataka", "mushti"]
TRANSITION_MUDRAS = ["suchi", "kartarimukha"]
ALL_MUDRAS = STATIC_MUDRAS + TRANSITION_MUDRAS

CLIP_TYPES = ["static", "transition"]

# --- MediaPipe Hands landmark layout -------------------------------------
# 21 landmarks per hand, each with (x, y, z). Index reference:
#   0 wrist
#   1-4   thumb (CMC, MCP, IP, TIP)
#   5-8   index (MCP, PIP, DIP, TIP)
#   9-12  middle
#   13-16 ring
#   17-20 pinky
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z

LANDMARK_COLUMNS = [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

METADATA_COLUMNS = [
    "mudra",
    "clip_type",
    "subject",
    "angle",
    "distance",
    "lighting",
    "clip_idx",
    "frame_idx",
    "timestamp_ms",
    "handedness",
]

ALL_COLUMNS = METADATA_COLUMNS + LANDMARK_COLUMNS

# Finger joint triplets used for joint-angle features in features.py.
# Each tuple is (a, b, c) landmark indices -> angle at b, between a-b and c-b.
FINGER_JOINT_TRIPLETS = {
    "thumb_mcp": (0, 1, 2),
    "thumb_ip": (1, 2, 3),
    "thumb_tip": (2, 3, 4),
    "index_mcp": (0, 5, 6),
    "index_pip": (5, 6, 7),
    "index_dip": (6, 7, 8),
    "middle_mcp": (0, 9, 10),
    "middle_pip": (9, 10, 11),
    "middle_dip": (10, 11, 12),
    "ring_mcp": (0, 13, 14),
    "ring_pip": (13, 14, 15),
    "ring_dip": (14, 15, 16),
    "pinky_mcp": (0, 17, 18),
    "pinky_pip": (17, 18, 19),
    "pinky_dip": (18, 19, 20),
}

# Fingertip indices, used for normalized inter-landmark distance features.
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
