from __future__ import annotations

import os
import tempfile
from pathlib import Path

mpl_config_dir = Path(tempfile.gettempdir()) / "health_monitor_ai_matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def write_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    path: str | Path,
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    Path(path).write_text(report, encoding="utf-8")


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    path: str | Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if plt is None:
        fallback_path = Path(path).with_suffix(".txt")
        fallback_path.write_text(
            "matplotlib is not installed; confusion matrix image was not generated.\n"
            f"class_names={class_names}\n"
            f"matrix={matrix.tolist()}\n",
            encoding="utf-8",
        )
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
