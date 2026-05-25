from __future__ import annotations

from pathlib import Path
from typing import Any

from src.radar.schema import RadarState


def _thresholds(config: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in config["rules"]["thresholds"].items()}


def _trust(config: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in config["rules"]["trust"].items()}


def generate_rules_header() -> str:
    states = "\n".join(f"    RADAR_STATE_{state.name} = {index}," for index, state in enumerate(RadarState))
    return f"""#ifndef RADAR_RULES_V1_H
#define RADAR_RULES_V1_H

#include <stdint.h>

typedef enum
{{
{states}
}} radar_state_t;

typedef struct
{{
    uint32_t min_sample_count;
    float quality_min;
    float invalid_fraction_max;
    float no_target_presence_max;
    float valid_presence_min;
    float stable_motion_max;
    float motion_min;
    float trust_valid_stable;
    float trust_valid_motion;
    float trust_no_target;
    float trust_radar_invalid;
}} radar_rules_v1_t;

extern const radar_rules_v1_t g_radar_rules_v1;

#endif /* RADAR_RULES_V1_H */
"""


def generate_rules_source(config: dict[str, Any], header_path: str | Path) -> str:
    thresholds = _thresholds(config)
    trust = _trust(config)
    header_name = Path(header_path).name
    return f"""#include "{header_name}"

const radar_rules_v1_t g_radar_rules_v1 = {{
    .min_sample_count = {int(thresholds["min_sample_count"])}u,
    .quality_min = {thresholds["quality_min"]:.6f}f,
    .invalid_fraction_max = {thresholds["invalid_fraction_max"]:.6f}f,
    .no_target_presence_max = {thresholds["no_target_presence_max"]:.6f}f,
    .valid_presence_min = {thresholds["valid_presence_min"]:.6f}f,
    .stable_motion_max = {thresholds["stable_motion_max"]:.6f}f,
    .motion_min = {thresholds["motion_min"]:.6f}f,
    .trust_valid_stable = {trust[RadarState.VALID_STABLE.value]:.6f}f,
    .trust_valid_motion = {trust[RadarState.VALID_MOTION.value]:.6f}f,
    .trust_no_target = {trust[RadarState.NO_TARGET.value]:.6f}f,
    .trust_radar_invalid = {trust[RadarState.RADAR_INVALID.value]:.6f}f,
}};
"""
