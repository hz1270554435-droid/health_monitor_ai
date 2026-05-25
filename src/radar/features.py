from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.radar.clean import INVALID_FRACTION_COLUMN, TIME_SECONDS_COLUMN


def _feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    configured = list(config.get("features", {}).get("columns", []))
    if not configured:
        configured = [
            "presence_score",
            "motion_score",
            "rr",
            "hr",
            "phase_amp",
            "phase_var",
            "quality_score",
        ]
    return [column for column in configured if column in frame.columns]


def _stat_value(values: pd.Series, stat: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if stat == "missing_fraction":
        return float(numeric.isna().mean()) if len(numeric) else 1.0
    if valid.empty:
        return float("nan")
    if stat == "mean":
        return float(valid.mean())
    if stat == "std":
        return float(valid.std(ddof=0))
    if stat == "min":
        return float(valid.min())
    if stat == "max":
        return float(valid.max())
    if stat == "median":
        return float(valid.median())
    if stat == "p10":
        return float(np.percentile(valid.to_numpy(dtype=float), 10))
    if stat == "p90":
        return float(np.percentile(valid.to_numpy(dtype=float), 90))
    if stat == "last":
        return float(valid.iloc[-1])
    if stat == "delta":
        return float(valid.iloc[-1] - valid.iloc[0])
    raise ValueError(f"Unsupported radar feature statistic: {stat}")


def _window_mask(frame: pd.DataFrame, window: pd.Series, group_columns: list[str]) -> pd.Series:
    mask = (frame[TIME_SECONDS_COLUMN] >= float(window["window_start"])) & (
        frame[TIME_SECONDS_COLUMN] < float(window["window_end"])
    )
    for column in group_columns:
        if column in window and column in frame.columns:
            mask &= frame[column].astype(str) == str(window[column])
    return mask


def extract_window_features(
    frame: pd.DataFrame,
    windows: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if windows.empty:
        raise ValueError("No radar windows available for feature extraction")
    if TIME_SECONDS_COLUMN not in frame.columns:
        raise ValueError(f"Cleaned radar frame must include {TIME_SECONDS_COLUMN}")

    feature_cols = _feature_columns(frame, config)
    if not feature_cols:
        raise ValueError("No configured radar feature columns are present in the cleaned frame")
    stats = list(config.get("features", {}).get("stats", ["mean", "std", "min", "max", "last"]))
    group_cols = [column for column in ("session_id", "person_id") if column in windows.columns and column in frame.columns]

    rows: list[dict[str, object]] = []
    for _, window in windows.iterrows():
        mask = _window_mask(frame, window, group_cols)
        segment = frame.loc[mask].sort_values(TIME_SECONDS_COLUMN)
        if segment.empty:
            continue
        row: dict[str, object] = {
            "window_id": window["window_id"],
            "window_index": int(window["window_index"]),
            "window_start": float(window["window_start"]),
            "window_end": float(window["window_end"]),
            "window_seconds": float(window["window_seconds"]),
            "sample_count": int(len(segment)),
            "invalid_fraction": float(segment.get(INVALID_FRACTION_COLUMN, pd.Series([0.0])).mean()),
        }
        for column in group_cols:
            row[column] = window[column]
        for column in feature_cols:
            for stat in stats:
                row[f"{column}_{stat}"] = _stat_value(segment[column], stat)
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("Radar feature extraction produced no rows")
    return output
