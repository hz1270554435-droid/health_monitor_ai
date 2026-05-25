from __future__ import annotations

from enum import Enum
from typing import Iterable

import pandas as pd


class RadarState(str, Enum):
    VALID_STABLE = "valid_stable"
    VALID_MOTION = "valid_motion"
    NO_TARGET = "no_target"
    RADAR_INVALID = "radar_invalid"


DEFAULT_REQUIRED_COLUMNS = ["timestamp", "presence_score", "motion_score"]
DEFAULT_OPTIONAL_COLUMNS = [
    "session_id",
    "person_id",
    "rr",
    "hr",
    "phase_amp",
    "phase_var",
    "quality_score",
]
DEFAULT_NUMERIC_COLUMNS = [
    "presence_score",
    "motion_score",
    "rr",
    "hr",
    "phase_amp",
    "phase_var",
    "quality_score",
]


def required_columns(config: dict) -> list[str]:
    return list(config.get("required_columns", DEFAULT_REQUIRED_COLUMNS))


def optional_columns(config: dict) -> list[str]:
    return list(config.get("optional_columns", DEFAULT_OPTIONAL_COLUMNS))


def numeric_columns(config: dict) -> list[str]:
    return list(config.get("numeric_columns", DEFAULT_NUMERIC_COLUMNS))


def group_columns(config: dict, frame: pd.DataFrame) -> list[str]:
    configured = list(config.get("group_columns", ["session_id", "person_id"]))
    return [column for column in configured if column in frame.columns]


def validate_required_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {', '.join(missing)}")


def validate_state_values(values: Iterable[object], context: str) -> None:
    allowed = {state.value for state in RadarState}
    invalid = sorted({str(value) for value in values if pd.notna(value) and str(value) not in allowed})
    if invalid:
        raise ValueError(f"{context} has invalid radar_state values: {invalid}; allowed={sorted(allowed)}")
