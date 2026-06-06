"""
Train the clustering-algorithm-selection classifier.

Reproduces the original notebook's training run:
    * read ``data/data_scaled.xlsx``
    * encode labels  KMeans=0, DBSCAN=1, OPTICS=2, GMM=3
    * 80/20 train/test split (``random_state=42``), 20% of train as validation
    * train for 100 epochs, evaluate on the held-out (synthetic) test split

Run from the repository root:
    python -m src.train

The reported accuracy is on a held-out split of the *synthetic* dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .model import BATCH_SIZE, EPOCHS, build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "data_scaled.xlsx"
MODEL_PATH = REPO_ROOT / "models" / "classifier.keras"
HISTORY_PATH = REPO_ROOT / "models" / "history.json"

LABEL_MAP = {"KMeans": 0, "DBSCAN": 1, "OPTICS": 2, "GMM": 3}


def load_xy(data_path: Path = DATA_PATH):
    """Load the scaled feature table and return (X, y) with encoded labels."""
    df = pd.read_excel(data_path)
    df["Best Algorithm"] = df["Best Algorithm"].map(LABEL_MAP)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return X, y


def main() -> None:
    import tensorflow as tf

    # Seed everything for a reproducible run (the report's results use seed 42).
    tf.keras.utils.set_random_seed(42)

    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )

    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy (synthetic test split): {accuracy * 100:.2f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history.history, f)
    print(f"Saved model to {MODEL_PATH} and history to {HISTORY_PATH}")


if __name__ == "__main__":
    main()
