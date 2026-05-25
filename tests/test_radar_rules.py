from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.radar.rules import apply_radar_rules


def _config() -> dict:
    return {
        "rules": {
            "columns": {
                "sample_count": "sample_count",
                "presence_score": "presence_score_mean",
                "motion_score": "motion_score_mean",
                "quality_score": "quality_score_mean",
                "invalid_fraction": "invalid_fraction",
            },
            "thresholds": {
                "min_sample_count": 3,
                "quality_min": 0.4,
                "invalid_fraction_max": 0.4,
                "no_target_presence_max": 0.2,
                "valid_presence_min": 0.5,
                "stable_motion_max": 0.35,
                "motion_min": 0.35,
            },
            "trust": {
                "valid_stable": 0.9,
                "valid_motion": 0.65,
                "no_target": 0.3,
                "radar_invalid": 0.0,
            },
        }
    }


def test_apply_radar_rules_outputs_four_states() -> None:
    features = pd.DataFrame(
        [
            {
                "sample_count": 5,
                "presence_score_mean": 0.8,
                "motion_score_mean": 0.1,
                "quality_score_mean": 1.0,
                "invalid_fraction": 0.0,
            },
            {
                "sample_count": 5,
                "presence_score_mean": 0.8,
                "motion_score_mean": 0.6,
                "quality_score_mean": 1.0,
                "invalid_fraction": 0.0,
            },
            {
                "sample_count": 5,
                "presence_score_mean": 0.1,
                "motion_score_mean": 0.1,
                "quality_score_mean": 1.0,
                "invalid_fraction": 0.0,
            },
            {
                "sample_count": 1,
                "presence_score_mean": 0.8,
                "motion_score_mean": 0.1,
                "quality_score_mean": 1.0,
                "invalid_fraction": 0.0,
            },
        ]
    )

    output = apply_radar_rules(features, _config())

    assert list(output["radar_state"]) == [
        "valid_stable",
        "valid_motion",
        "no_target",
        "radar_invalid",
    ]
    assert output.loc[0, "trust_factor"] > output.loc[2, "trust_factor"]
