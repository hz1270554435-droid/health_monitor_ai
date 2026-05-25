from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def state_summary(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if prediction_column not in frame.columns:
        raise ValueError(f"Prediction column not found: {prediction_column}")
    counts = frame[prediction_column].astype(str).value_counts().sort_index()
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "state_counts": {str(key): int(value) for key, value in counts.items()},
    }
    if "trust_factor" in frame.columns:
        summary["trust_factor_mean"] = float(pd.to_numeric(frame["trust_factor"], errors="coerce").mean())
    if "radar_quality_score" in frame.columns:
        summary["radar_quality_score_mean"] = float(
            pd.to_numeric(frame["radar_quality_score"], errors="coerce").mean()
        )
    return summary


def evaluate_state_predictions(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
) -> dict[str, Any]:
    metrics = state_summary(frame, prediction_column)
    if target_column not in frame.columns:
        metrics["supervised"] = False
        metrics["details"] = f"Target column not found: {target_column}"
        return metrics

    rows = frame[[target_column, prediction_column]].dropna()
    if rows.empty:
        metrics["supervised"] = False
        metrics["details"] = "No non-empty target/prediction rows available"
        return metrics

    y_true = rows[target_column].astype(str)
    y_pred = rows[prediction_column].astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics.update(
        {
            "supervised": True,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "labels": labels,
            "confusion_matrix": matrix.astype(int).tolist(),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }
    )
    return metrics
