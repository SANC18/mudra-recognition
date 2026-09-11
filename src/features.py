"""
Feature engineering shared by the static classifier (Phase 2) and the
sequence model (Phase 3).

Raw (x, y) landmark positions are NOT scale- or rotation-invariant: the same
mudra recorded closer to the camera, or with the hand rotated, produces very
different raw numbers. Instead we compute:

  1. Joint angles at each knuckle (invariant to hand position/scale/rotation
     in-plane).
  2. Inter-landmark distances between fingertips and the wrist, normalized by
     a hand-size reference (invariant to distance-from-camera).

Both operate on a single frame's landmarks: a (21, 3) array of (x, y, z)
per landmark, in the same order MediaPipe returns them (see config.py).
"""

import numpy as np
import pandas as pd

from config import FINGER_JOINT_TRIPLETS, FINGERTIPS, LANDMARK_COLUMNS


def row_to_landmark_array(row: pd.Series) -> np.ndarray:
    """Convert one CSV row's flattened x0,y0,z0,...,x20,y20,z20 columns into a (21, 3) array."""
    coords = row[LANDMARK_COLUMNS].to_numpy(dtype=float)
    return coords.reshape(21, 3)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in radians between two vectors, robust to zero-length vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def compute_joint_angles(landmarks: np.ndarray) -> dict:
    """Angle at each knuckle/joint defined in FINGER_JOINT_TRIPLETS. Rotation/scale invariant."""
    angles = {}
    for name, (a, b, c) in FINGER_JOINT_TRIPLETS.items():
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        angles[f"angle_{name}"] = _angle_between(v1, v2)
    return angles


def compute_normalized_distances(landmarks: np.ndarray) -> dict:
    """
    Distance from each fingertip to the wrist, normalized by a hand-size
    reference (wrist-to-middle-MCP distance) so it's invariant to how close
    the hand is to the camera.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_size = np.linalg.norm(middle_mcp - wrist)
    hand_size = hand_size if hand_size > 1e-6 else 1e-6  # avoid divide-by-zero

    distances = {}
    for name, tip_idx in FINGERTIPS.items():
        raw_dist = np.linalg.norm(landmarks[tip_idx] - wrist)
        distances[f"dist_{name}_tip_wrist"] = raw_dist / hand_size

    # Fingertip-to-fingertip distances (useful for shapes like Suchi/Kartarimukha
    # where the relationship between two specific fingertips is what differs).
    tips = list(FINGERTIPS.items())
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            name_i, idx_i = tips[i]
            name_j, idx_j = tips[j]
            raw_dist = np.linalg.norm(landmarks[idx_i] - landmarks[idx_j])
            distances[f"dist_{name_i}_{name_j}_tip"] = raw_dist / hand_size

    return distances


def extract_feature_vector(landmarks: np.ndarray) -> dict:
    """Full feature dict for one frame: joint angles + normalized distances."""
    features = {}
    features.update(compute_joint_angles(landmarks))
    features.update(compute_normalized_distances(landmarks))
    return features


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply extract_feature_vector to every row of a raw landmark DataFrame
    (as produced by extract_landmarks.py) and return a new DataFrame with the
    engineered features alongside the original metadata columns.
    """
    metadata_cols = [c for c in df.columns if c not in LANDMARK_COLUMNS]
    feature_rows = []

    for _, row in df.iterrows():
        landmarks = row_to_landmark_array(row)
        feature_rows.append(extract_feature_vector(landmarks))

    features_df = pd.DataFrame(feature_rows)
    return pd.concat(
        [df[metadata_cols].reset_index(drop=True), features_df], axis=1
    )
