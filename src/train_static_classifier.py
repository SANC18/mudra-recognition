"""
Phase 2 — Static mudra classifier.

Loads extracted landmark CSVs for the static mudras, engineers scale/rotation
invariant features (features.py), and trains + compares a Random Forest, an
SVM, and a small MLP. Reports per-mudra accuracy and a confusion matrix for
the best model.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LANDMARKS_DIR, MODELS_DIR, STATIC_MUDRAS  # noqa: E402
from features import build_feature_dataframe  # noqa: E402


def load_static_data() -> pd.DataFrame:
    """Load and concatenate landmark CSVs for all static mudras, keeping only static-clip rows."""
    frames = []
    for mudra in STATIC_MUDRAS:
        csv_path = LANDMARKS_DIR / f"{mudra}.csv"
        if not csv_path.exists():
            print(f"  [warn] {csv_path} not found — run extract_landmarks.py first")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["clip_type"] == "static"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No static mudra landmark CSVs found in data/landmarks/. "
            "Run src/extract_landmarks.py first."
        )
    return pd.concat(frames, ignore_index=True)


def train_and_evaluate(features_df: pd.DataFrame):
    feature_cols = [
        c for c in features_df.columns
        if c.startswith("angle_") or c.startswith("dist_")
    ]
    X = features_df[feature_cols].to_numpy()
    y = features_df["mudra"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # RF is scale-invariant; SVM/MLP benefit from the scaled features.
        if name == "random_forest":
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "model": model,
            "accuracy": report["accuracy"],
            "y_pred": y_pred,
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred, zero_division=0))

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

    cm = confusion_matrix(y_test, results[best_name]["y_pred"], labels=sorted(set(y)))
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(cm, index=sorted(set(y)), columns=sorted(set(y))))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["model"], MODELS_DIR / "static_classifier.joblib")
    joblib.dump(scaler, MODELS_DIR / "static_scaler.joblib")
    print(f"\nSaved best model to {MODELS_DIR / 'static_classifier.joblib'}")


def main():
    print("Loading static mudra landmark data...")
    raw_df = load_static_data()
    print(f"Loaded {len(raw_df)} hand-frames across {raw_df['mudra'].nunique()} mudras")

    print("Engineering features...")
    features_df = build_feature_dataframe(raw_df)

    print("Training and evaluating models...")
    train_and_evaluate(features_df)


if __name__ == "__main__":
    main()
