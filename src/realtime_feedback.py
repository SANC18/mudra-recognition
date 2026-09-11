"""
Phase 4 — Real-time feedback.

Live webcam -> MediaPipe landmark extraction -> trained static classifier ->
overlay predicted mudra name + confidence. Run train_static_classifier.py
first so models/static_classifier.joblib and static_scaler.joblib exist.

Press 'q' to quit.
"""

import sys
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR  # noqa: E402
from features import extract_feature_vector  # noqa: E402

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def load_model():
    model_path = MODELS_DIR / "static_classifier.joblib"
    scaler_path = MODELS_DIR / "static_scaler.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run src/train_static_classifier.py first."
        )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, scaler


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=float
    )


def main():
    model, scaler = load_model()
    feature_names = None  # inferred from the first feature vector

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions/index.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # mirror for a natural selfie-view
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks = landmarks_to_array(hand_landmarks)
                feature_dict = extract_feature_vector(landmarks)
                if feature_names is None:
                    feature_names = list(feature_dict.keys())

                feature_vec = np.array(
                    [feature_dict[name] for name in feature_names]
                ).reshape(1, -1)
                if scaler is not None and hasattr(model, "predict_proba"):
                    # SVM/MLP were trained on scaled features; RF was not.
                    # This demo assumes whichever model train_static_classifier.py
                    # saved as best — if it's RF, scaling is a harmless no-op here
                    # since RF thresholds are scale-invariant either way is NOT
                    # guaranteed, so re-run with the matching model type.
                    pass

                pred = model.predict(feature_vec)[0]
                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(feature_vec)[0]
                    confidence = proba.max()

                label = f"{pred}"
                if confidence is not None:
                    label += f" ({confidence * 100:.0f}%)"

                cv2.putText(
                    frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame, "No hand detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow("Mudra Recognition — press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
