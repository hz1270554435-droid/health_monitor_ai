from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.radar.clean import clean_radar_frame
from src.radar.features import extract_window_features
from src.radar.windows import make_window_table


def test_extract_window_features_computes_expected_stats() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4],
            "session_id": ["S1"] * 5,
            "person_id": ["P1"] * 5,
            "presence_score": [0.1, 0.3, 0.5, 0.7, 0.9],
            "motion_score": [0.2, 0.2, 0.4, 0.4, 0.6],
        }
    )
    radar_config = {
        "timestamp_column": "timestamp",
        "required_columns": ["timestamp", "presence_score", "motion_score"],
        "numeric_columns": ["presence_score", "motion_score"],
        "group_columns": ["session_id", "person_id"],
        "window_seconds": 5.0,
        "hop_seconds": 2.5,
        "min_samples_per_window": 3,
    }
    feature_config = {
        "features": {
            "columns": ["presence_score", "motion_score"],
            "stats": ["mean", "max", "last", "delta", "missing_fraction"],
        }
    }

    cleaned, _ = clean_radar_frame(frame, radar_config)
    windows = make_window_table(cleaned, radar_config)
    features = extract_window_features(cleaned, windows, feature_config)

    first = features.iloc[0]
    assert first["presence_score_mean"] == 0.5
    assert first["presence_score_max"] == 0.9
    assert first["presence_score_delta"] == 0.8
    assert first["motion_score_missing_fraction"] == 0.0
