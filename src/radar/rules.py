from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.radar.schema import RadarState


def _value(row: pd.Series, column: str, default: float) -> float:
    if column not in row or pd.isna(row[column]):
        return default
    return float(row[column])


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _quality_score(
    sample_count: float,
    quality_source: float,
    invalid_fraction: float,
    thresholds: dict[str, Any],
) -> float:
    min_samples = max(float(thresholds["min_sample_count"]), 1.0)
    invalid_max = max(float(thresholds["invalid_fraction_max"]), 1e-6)
    sample_factor = _clip01(sample_count / min_samples)
    invalid_factor = _clip01(1.0 - invalid_fraction / invalid_max)
    return _clip01(sample_factor * _clip01(quality_source) * invalid_factor)


def classify_feature_row(row: pd.Series, rules_config: dict[str, Any]) -> dict[str, object]:
    columns = rules_config["columns"]
    thresholds = rules_config["thresholds"]
    trust = rules_config["trust"]

    sample_count = _value(row, columns["sample_count"], 0.0)
    presence = _clip01(_value(row, columns["presence_score"], 0.0))
    motion = _clip01(_value(row, columns["motion_score"], 0.0))
    quality_source = _clip01(_value(row, columns.get("quality_score", ""), 1.0))
    invalid_fraction = _clip01(_value(row, columns["invalid_fraction"], 0.0))
    radar_quality = _quality_score(sample_count, quality_source, invalid_fraction, thresholds)

    if sample_count < float(thresholds["min_sample_count"]):
        state = RadarState.RADAR_INVALID
        reason = "too_few_samples"
    elif invalid_fraction > float(thresholds["invalid_fraction_max"]):
        state = RadarState.RADAR_INVALID
        reason = "invalid_fraction_high"
    elif radar_quality < float(thresholds["quality_min"]):
        state = RadarState.RADAR_INVALID
        reason = "quality_low"
    elif presence <= float(thresholds["no_target_presence_max"]):
        state = RadarState.NO_TARGET
        reason = "presence_low"
    elif presence >= float(thresholds["valid_presence_min"]) and motion >= float(thresholds["motion_min"]):
        state = RadarState.VALID_MOTION
        reason = "motion_high"
    elif presence >= float(thresholds["valid_presence_min"]) and motion < float(thresholds["stable_motion_max"]):
        state = RadarState.VALID_STABLE
        reason = "presence_stable"
    else:
        state = RadarState.RADAR_INVALID
        reason = "uncertain_presence_motion"

    base_trust = float(trust[state.value])
    return {
        "radar_state": state.value,
        "radar_quality_score": radar_quality,
        "presence_score": presence,
        "motion_score": motion,
        "trust_factor": _clip01(base_trust * radar_quality),
        "rule_reason": reason,
    }


def apply_radar_rules(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rules_config = config["rules"]
    required = [
        rules_config["columns"]["sample_count"],
        rules_config["columns"]["presence_score"],
        rules_config["columns"]["motion_score"],
        rules_config["columns"]["invalid_fraction"],
    ]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Radar feature table is missing rule input columns: {', '.join(missing)}")

    decisions = [classify_feature_row(row, rules_config) for _, row in features.iterrows()]
    decision_frame = pd.DataFrame(decisions)
    return pd.concat([features.reset_index(drop=True), decision_frame], axis=1)
