"""
Recording helper for Phase 1.

Interactively records a webcam clip and saves it directly into
data/raw_videos/<mudra>/<clip_type>/ using the naming convention from
docs/mudras.md — so you never have to manually rename or move files.

Usage:
    python src/record_clip.py

Controls while the preview window is focused:
    r  - start / stop recording
    q  - quit
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ALL_MUDRAS, RAW_VIDEOS_DIR, STATIC_MUDRAS, TRANSITION_MUDRAS  # noqa: E402


def prompt_choice(label: str, options: list[str]) -> str:
    print(f"\n{label}:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        choice = input(f"Enter number (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, try again.")


def next_clip_idx(mudra: str, clip_type: str, subject: str, angle: str, distance: str, lighting: str) -> str:
    """Find the next free clip index for this exact condition combo."""
    folder = RAW_VIDEOS_DIR / mudra / clip_type
    folder.mkdir(parents=True, exist_ok=True)
    prefix = f"{subject}__{angle}__{distance}__{lighting}__"
    existing = [p.stem for p in folder.glob(f"{prefix}*.mp4")]
    used_indices = set()
    for stem in existing:
        idx_str = stem.split("__")[-1]
        if idx_str.isdigit():
            used_indices.add(int(idx_str))
    idx = 1
    while idx in used_indices:
        idx += 1
    return f"{idx:02d}"


def record(output_path: Path, fps: int = 20):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions/index.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = None
    recording = False

    print("\nPreview window open. Press 'r' to start recording, 'r' again to stop, 'q' to quit without saving further.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        display = frame.copy()
        status_text = "RECORDING" if recording else "Press 'r' to start"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.imshow(f"Recording -> {output_path.name} (press 'r' start/stop, 'q' quit)", display)

        if recording and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            if not recording:
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                recording = True
                print("  -> recording started")
            else:
                recording = False
                writer.release()
                writer = None
                print(f"  -> recording stopped, saved to {output_path}")
                break
        elif key == ord("q"):
            print("  -> quit without finishing this clip")
            if writer is not None:
                writer.release()
                if output_path.exists():
                    output_path.unlink()  # discard incomplete clip
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=== Mudra clip recorder ===")

    while True:
        mudra = prompt_choice("Mudra", ALL_MUDRAS)
        clip_type = "static" if mudra in STATIC_MUDRAS else "transition"
        print(f"(clip_type auto-set to '{clip_type}' for {mudra})")

        subject = input("Subject id [me]: ").strip() or "me"
        angle = prompt_choice("Angle", ["front", "left30", "right30", "above", "below"])
        distance = prompt_choice("Distance", ["close", "medium", "far"])
        lighting = prompt_choice("Lighting", ["bright", "dim"])

        clip_idx = next_clip_idx(mudra, clip_type, subject, angle, distance, lighting)
        filename = f"{subject}__{angle}__{distance}__{lighting}__{clip_idx}.mp4"
        output_path = RAW_VIDEOS_DIR / mudra / clip_type / filename

        print(f"\nAbout to record: {output_path}")
        if clip_type == "transition":
            print("Remember: start from a neutral hand, transition into the shape, then hold.")
        input("Press Enter when ready to open the camera preview...")

        record(output_path)

        again = input("\nRecord another clip? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nDone. Run 'python src/extract_landmarks.py' once you've recorded your pilot clips.")


if __name__ == "__main__":
    main()
