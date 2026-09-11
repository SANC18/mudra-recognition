# Mudra Recognition & Correction System

Real-time recognition and correction of classical Indian dance hand mudras from webcam
input. Classifies both static (held) poses and dynamic transitions, and gives live
feedback on form. Built entirely in Python on a self-recorded landmark dataset — no
public dataset exists for this task.

## Why this project

- No public dataset exists for classical mudra recognition — the dataset itself
  (collection + labeling) is original work.
- Combines classical ML (landmark-based classifiers) with sequence modeling
  (LSTM/GRU) rather than being a single-technique demo.
- Produces a real-time, demo-able artifact, not just a static repo.

## Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Dataset collection — self-recorded video, MediaPipe Hands landmark extraction | In progress |
| 2 | Static mudra classifier — joint-angle/distance features, RF/SVM/MLP | Not started |
| 3 | Dynamic sequence modeling — LSTM/GRU over landmark sequences | Not started |
| 4 | Real-time feedback — live webcam overlay of prediction + confidence | Not started |
| 5 | Writeup — dataset description, model comparison, accuracy/confusion matrices (optional) | Not started |

## Starter mudra set (Phase 1)

| Mudra | Type |
|---|---|
| Pataka | Static |
| Tripataka | Static |
| Mushti | Static |
| Suchi | Transition |
| Kartarimukha | Transition |

See `docs/mudras.md` for descriptions, references, and the recording protocol.

## Project structure

```
mudra-recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw_videos/       # gitignored — self-recorded clips, kept local only
│   └── landmarks/        # tracked — extracted structured landmark data (CSV)
├── src/
│   ├── config.py             # mudra labels, paths, constants
│   ├── extract_landmarks.py  # Phase 1: video -> landmark CSV via MediaPipe
│   ├── features.py           # Phase 2/3: scale/rotation-invariant feature engineering
│   ├── train_static_classifier.py   # Phase 2: RF/SVM/MLP on single-frame features
│   ├── train_sequence_model.py      # Phase 3: LSTM/GRU on landmark sequences
│   └── realtime_feedback.py         # Phase 4: live webcam demo
├── notebooks/            # exploration / pilot testing
└── docs/
    └── mudras.md         # mudra reference list, labels, recording protocol
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage (as phases are completed)

```bash
# Phase 1 — extract landmarks from a folder of recorded clips
python src/extract_landmarks.py

# Phase 2 — train and evaluate the static classifier
python src/train_static_classifier.py

# Phase 3 — train the sequence model for transition mudras
python src/train_sequence_model.py

# Phase 4 — run the real-time webcam demo
python src/realtime_feedback.py
```
