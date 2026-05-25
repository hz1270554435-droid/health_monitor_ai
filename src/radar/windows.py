from __future__ import annotations

from typing import Any

import pandas as pd

from src.radar.clean import TIME_SECONDS_COLUMN
from src.radar.schema import group_columns


def _window_id(group_values: dict[str, object], index: int) -> str:
    prefix_parts = [str(value) for value in group_values.values() if pd.notna(value)]
    prefix = "_".join(prefix_parts) if prefix_parts else "radar"
    safe_prefix = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prefix)
    return f"{safe_prefix}_{index:05d}"


def make_window_table(frame: pd.DataFrame, radar_config: dict[str, Any]) -> pd.DataFrame:
    if TIME_SECONDS_COLUMN not in frame.columns:
        raise ValueError(f"Cleaned radar frame must include {TIME_SECONDS_COLUMN}")

    window_seconds = float(radar_config.get("window_seconds", 5.0))
    hop_seconds = float(radar_config.get("hop_seconds", 2.5))
    min_samples = int(radar_config.get("min_samples_per_window", 1))
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if min_samples <= 0:
        raise ValueError("min_samples_per_window must be positive")

    group_cols = group_columns(radar_config, frame)
    grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, dropna=False, sort=True)
    rows: list[dict[str, object]] = []

    for group_key, group in grouped:
        group_frame = group.sort_values(TIME_SECONDS_COLUMN)
        start_time = float(group_frame[TIME_SECONDS_COLUMN].min())
        end_limit = float(group_frame[TIME_SECONDS_COLUMN].max())
        if isinstance(group_key, tuple):
            group_values = dict(zip(group_cols, group_key))
        else:
            group_values = dict(zip(group_cols, (group_key,)))

        index = 0
        current = start_time
        while current <= end_limit:
            window_end = current + window_seconds
            mask = (group_frame[TIME_SECONDS_COLUMN] >= current) & (group_frame[TIME_SECONDS_COLUMN] < window_end)
            sample_count = int(mask.sum())
            if sample_count >= min_samples:
                row: dict[str, object] = {
                    **group_values,
                    "window_id": _window_id(group_values, index),
                    "window_index": index,
                    "window_start": current,
                    "window_end": window_end,
                    "window_seconds": window_seconds,
                    "sample_count": sample_count,
                }
                rows.append(row)
            index += 1
            current = start_time + index * hop_seconds

    return pd.DataFrame(rows)
