"""
Evaluate a trained classifier on the held-out (synthetic) test split.

Reproduces the same 80/20 split as training (``random_state=42``), then prints
accuracy, a classification report, and a confusion matrix. Optionally writes the
confusion matrix to ``figures/confusion_matrix.png``.

Run from the repository root (after ``python -m src.train``):
    python -m src.evaluate

All metrics are on synthetic data; they do not measure real-world performance.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from .model import CLASS_NAMES
from .train import DATA_PATH, MODEL_PATH, load_xy

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_PATH = REPO_ROOT / "figures" / "confusion_matrix.png"


def main(save_figure: bool = True) -> None:
    import numpy as np
    from tensorflow.keras.models import load_model

    X, y = load_xy(DATA_PATH)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    labels = [0, 1, 2, 3]
    names = [CLASS_NAMES[i] for i in labels]

    print(f"Accuracy (synthetic test split): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=labels, target_names=names))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if save_figure:
        import matplotlib.pyplot as plt

        FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title("Confusion matrix (synthetic test split)")
        plt.tight_layout()
        plt.savefig(FIG_PATH, dpi=150)
        print(f"Saved confusion matrix to {FIG_PATH}")


if __name__ == "__main__":
    main()
