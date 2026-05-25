from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.radar.schema import numeric_columns, required_columns, validate_required_columns


TIME_SECONDS_COLUMN = "_time_seconds"
INVALID_FRACTION_COLUMN = "invalid_fraction"


def load_radar_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Parsed radar CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Parsed radar CSV has no rows: {csv_path}")
    return frame


def timestamp_to_seconds(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and numeric.notna().sum() >= series.notna().sum() * 0.8:
        return numeric.astype(float)

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError("timestamp column cannot be parsed as numeric seconds or datetimes")
    origin = parsed.dropna().min()
    return (parsed - origin).dt.total_seconds()


def clean_radar_frame(frame: pd.DataFrame, radar_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    timestamp_column = str(radar_config.get("timestamp_column", "timestamp"))
    validate_required_columns(frame, required_columns(radar_config), "radar CSV")

    output = frame.copy()
    output[TIME_SECONDS_COLUMN] = timestamp_to_seconds(output[timestamp_column])
    invalid_timestamp_count = int(output[TIME_SECONDS_COLUMN].isna().sum())
    output = output[output[TIME_SECONDS_COLUMN].notna()].copy()
    if output.empty:
        raise ValueError("No rows remain after dropping invalid timestamps")

    report: dict[str, Any] = {
        "input_rows": int(len(frame)),
        "invalid_timestamp_count": invalid_timestamp_count,
        "numeric_columns": {},
        "clip_ranges": {},
    }

    present_numeric = [column for column in numeric_columns(radar_config) if column in output.columns]
    for column in present_numeric:
        before_missing = int(output[column].isna().sum())
        output[column] = pd.to_numeric(output[column], errors="coerce")
        after_missing = int(output[column].isna().sum())
        report["numeric_columns"][column] = {
            "missing_before_numeric": before_missing,
            "missing_after_numeric": after_missing,
        }

    clip_ranges = dict(radar_config.get("clip_ranges", {}))
    for column, bounds in clip_ranges.items():
        if column not in output.columns:
            continue
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise ValueError(f"Invalid clip range for {column}: expected [min, max]")
        low, high = float(bounds[0]), float(bounds[1])
        values = output[column]
        outlier_mask = values.notna() & ((values < low) | (values > high))
        output.loc[outlier_mask, column] = np.nan
        report["clip_ranges"][column] = {
            "min": low,
            "max": high,
            "outlier_count": int(outlier_mask.sum()),
        }

    if present_numeric:
        output[INVALID_FRACTION_COLUMN] = output[present_numeric].isna().mean(axis=1).astype(float)
    else:
        output[INVALID_FRACTION_COLUMN] = 0.0

    sort_columns = [column for column in ("session_id", "person_id") if column in output.columns]
    sort_columns.append(TIME_SECONDS_COLUMN)
    output = output.sort_values(sort_columns).reset_index(drop=True)
    report["output_rows"] = int(len(output))
    report["time_start_seconds"] = float(output[TIME_SECONDS_COLUMN].min())
    report["time_end_seconds"] = float(output[TIME_SECONDS_COLUMN].max())
    return output, report
