from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.radar.clean import clean_radar_frame
from src.radar.windows import make_window_table


def test_make_window_table_uses_grouped_time_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4, 5, 6],
            "session_id": ["S1"] * 7,
            "person_id": ["P1"] * 7,
            "presence_score": [0.8] * 7,
            "motion_score": [0.1] * 7,
        }
    )
    config = {
        "timestamp_column": "timestamp",
        "required_columns": ["timestamp", "presence_score", "motion_score"],
        "numeric_columns": ["presence_score", "motion_score"],
        "group_columns": ["session_id", "person_id"],
        "window_seconds": 5.0,
        "hop_seconds": 2.5,
        "min_samples_per_window": 2,
    }
    cleaned, _ = clean_radar_frame(frame, config)
    windows = make_window_table(cleaned, config)

    assert list(windows["window_start"]) == [0.0, 2.5, 5.0]
    assert list(windows["sample_count"]) == [5, 4, 2]
    assert set(windows["person_id"]) == {"P1"}
