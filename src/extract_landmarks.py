"""
Phase 1 — Dataset collection.

Walks data/raw_videos/<mudra>/<clip_type>/<subject>__<angle>__<distance>__<lighting>__<idx>.mp4,
runs MediaPipe Hands over every frame, and writes one structured CSV per mudra to
data/landmarks/<mudra>.csv (one row per detected-hand-frame).

See docs/mudras.md for the filename convention this script expects.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_COLUMNS, LANDMARKS_DIR, RAW_VIDEOS_DIR  # noqa: E402

mp_hands = mp.solutions.hands


def parse_filename(video_path: Path, mudra: str, clip_type: str) -> dict:
    """Parse the `<subject>__<angle>__<distance>__<lighting>__<idx>.mp4` convention."""
    stem = video_path.stem
    parts = stem.split("__")
    if len(parts) != 5:
        raise ValueError(
            f"Filename '{video_path.name}' doesn't match the expected "
            "subject__angle__distance__lighting__idx.mp4 convention "
            "(see docs/mudras.md)."
        )
    subject, angle, distance, lighting, clip_idx = parts
    return {
        "mudra": mudra,
        "clip_type": clip_type,
        "subject": subject,
        "angle": angle,
        "distance": distance,
        "lighting": lighting,
        "clip_idx": clip_idx,
    }


def extract_from_video(video_path: Path, metadata: dict, hands) -> list[dict]:
    """Run MediaPipe Hands over one video, return one row dict per detected hand-frame."""
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.multi_hand_landmarks, result.multi_handedness
            ):
                row = dict(metadata)
                row["frame_idx"] = frame_idx
                row["timestamp_ms"] = timestamp_ms
                row["handedness"] = handedness.classification[0].label

                for i, lm in enumerate(hand_landmarks.landmark):
                    row[f"x{i}"] = lm.x
                    row[f"y{i}"] = lm.y
                    row[f"z{i}"] = lm.z

                rows.append(row)

        frame_idx += 1

    cap.release()
    return rows


def process_mudra(mudra: str) -> pd.DataFrame:
    """Process every clip (static + transition) for one mudra, return a combined DataFrame."""
    all_rows = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        for clip_type_dir in (RAW_VIDEOS_DIR / mudra).glob("*"):
            if not clip_type_dir.is_dir():
                continue
            clip_type = clip_type_dir.name

            video_paths = sorted(
                p
                for p in clip_type_dir.iterdir()
                if p.suffix.lower() in (".mp4", ".mov", ".avi")
            )
            if not video_paths:
                print(f"  [warn] no clips found in {clip_type_dir}")
                continue

            for video_path in video_paths:
                try:
                    metadata = parse_filename(video_path, mudra, clip_type)
                except ValueError as e:
                    print(f"  [skip] {e}")
                    continue

                rows = extract_from_video(video_path, metadata, hands)
                print(f"  {video_path.name}: {len(rows)} hand-frames extracted")
                all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=ALL_COLUMNS)
    return pd.DataFrame(all_rows)[ALL_COLUMNS]


def main():
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_VIDEOS_DIR.exists():
        print(f"No raw videos found at {RAW_VIDEOS_DIR} — record clips first "
              "(see docs/mudras.md for the folder/filename convention).")
        return

    mudra_dirs = sorted(p for p in RAW_VIDEOS_DIR.iterdir() if p.is_dir())
    if not mudra_dirs:
        print(f"{RAW_VIDEOS_DIR} exists but is empty — nothing to extract yet.")
        return

    for mudra_dir in mudra_dirs:
        mudra = mudra_dir.name
        print(f"Processing mudra: {mudra}")
        df = process_mudra(mudra)

        if df.empty:
            print(f"  [warn] no data extracted for '{mudra}', skipping CSV write")
            continue

        out_path = LANDMARKS_DIR / f"{mudra}.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
