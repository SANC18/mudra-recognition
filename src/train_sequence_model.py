"""
Phase 3 — Dynamic sequence modeling.

Models the full gesture-in-motion for transition mudras (e.g. Suchi,
Kartarimukha) using an LSTM over per-frame engineered features, so the model
classifies the transition rather than a single snapshot. Compares final
accuracy against the Phase 2 static-only baseline (reported separately by
train_static_classifier.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LANDMARKS_DIR, MODELS_DIR, TRANSITION_MUDRAS  # noqa: E402
from features import build_feature_dataframe  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transition_data() -> pd.DataFrame:
    frames = []
    for mudra in TRANSITION_MUDRAS:
        csv_path = LANDMARKS_DIR / f"{mudra}.csv"
        if not csv_path.exists():
            print(f"  [warn] {csv_path} not found — run extract_landmarks.py first")
            continue
        df = pd.read_csv(csv_path)
        df = df[df["clip_type"] == "transition"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No transition mudra landmark CSVs found in data/landmarks/. "
            "Run src/extract_landmarks.py first."
        )
    return pd.concat(frames, ignore_index=True)


def build_sequences(features_df: pd.DataFrame, feature_cols: list[str]):
    """
    Group frames into per-clip sequences (each clip = one continuous
    neutral -> shape recording), ordered by frame_idx.
    """
    group_cols = ["mudra", "subject", "angle", "distance", "lighting", "clip_idx"]
    sequences, labels = [], []

    for _, group in features_df.groupby(group_cols):
        group = group.sort_values("frame_idx")
        sequences.append(group[feature_cols].to_numpy(dtype=np.float32))
        labels.append(group["mudra"].iloc[0])

    return sequences, labels


class MudraSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.stack(lengths)
    labels = torch.stack(labels)
    return padded, lengths, labels


class MudraLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.classifier(last_hidden)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, lengths, y in loader:
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


def main(num_epochs: int = 30, batch_size: int = 8, hidden_size: int = 64, lr: float = 1e-3):
    print("Loading transition mudra landmark data...")
    raw_df = load_transition_data()

    print("Engineering features...")
    features_df = build_feature_dataframe(raw_df)
    feature_cols = [
        c for c in features_df.columns
        if c.startswith("angle_") or c.startswith("dist_")
    ]

    print("Building per-clip sequences...")
    sequences, labels = build_sequences(features_df, feature_cols)
    print(f"Built {len(sequences)} sequences across {len(set(labels))} mudras")

    # Normalize features across all frames (fit on flattened sequence data).
    scaler = StandardScaler()
    all_frames = np.concatenate(sequences, axis=0)
    scaler.fit(all_frames)
    sequences = [scaler.transform(s) for s in sequences]

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
    )

    train_ds = MudraSequenceDataset(seq_train, y_train)
    test_ds = MudraSequenceDataset(seq_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MudraLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_classes=len(label_encoder.classes_),
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_size": len(feature_cols),
            "hidden_size": hidden_size,
            "num_classes": len(label_encoder.classes_),
            "classes": label_encoder.classes_.tolist(),
            "feature_cols": feature_cols,
        },
        MODELS_DIR / "sequence_model.pt",
    )
    print(f"\nSaved model to {MODELS_DIR / 'sequence_model.pt'}")


if __name__ == "__main__":
    main()
