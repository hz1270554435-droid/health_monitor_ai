# Radar Assist Pipeline

## Role

Radar is now a MIC trust assistant. The MIC model remains the primary
`cough` / `non_cough` classifier. Radar does not directly diagnose cough, heart
rate abnormality, or respiration abnormality in v1.

Radar v1 uses parsed radar CSV only. It must not train on UART raw bytes.

## Why Rules First

The first radar_assist baseline is a rule state machine because it is:

- easy to inspect on small parsed CSV files
- easy to export to board C code
- stable before enough labeled radar data exists
- useful as a trust/reference signal for MIC fusion

The second stage can train a Random Forest on the same `radar_features_v1.csv`
feature table. XGBoost can be added later only if the dependency policy changes.

## States

| State | Meaning |
| --- | --- |
| `valid_stable` | Target is present, radar quality is acceptable, and motion is low/stable. |
| `valid_motion` | Target is present and radar quality is acceptable, but motion is high. |
| `no_target` | Radar confidence indicates no reliable target in the window. |
| `radar_invalid` | Window is too sparse, too noisy, missing too much data, or otherwise uncertain. |

## Parsed CSV Input

Configured by `configs/radar/radar_features_v1.yaml`.

Required columns:

- `timestamp`
- `presence_score`
- `motion_score`

Optional columns:

- `session_id`
- `person_id`
- `rr`
- `hr`
- `phase_amp`
- `phase_var`
- `quality_score`

`person_id` must be preserved when available. Any train/val/test split for the
future tree baseline must be grouped by `person_id`.

## Feature Table

Output: `data/processed/radar_assist_v1/radar_features_v1.csv`

This file is generated and should not be committed. It is reproducible from:

- parsed radar CSV
- `configs/radar/radar_features_v1.yaml`

Core metadata fields:

- `window_id`
- `window_index`
- `session_id`
- `person_id`
- `window_start`
- `window_end`
- `window_seconds`
- `sample_count`
- `invalid_fraction`

Feature columns follow this pattern:

```text
<radar_column>_<stat>
```

Examples:

- `presence_score_mean`
- `presence_score_p10`
- `motion_score_mean`
- `motion_score_delta`
- `quality_score_mean`
- `rr_median`
- `phase_var_p90`

The default v1 window is 5 s with 2.5 s hop.

## Rule Output

Output: `data/processed/radar_assist_v1/radar_rules_v1.csv`

Rule columns:

- `radar_state`
- `radar_quality_score`
- `presence_score`
- `motion_score`
- `trust_factor`
- `rule_reason`

`trust_factor` is meant for fusion. It is not a cough probability.

## Pipeline Commands

Inspect parsed radar CSV:

```powershell
python scripts/radar/inspect_radar_csv.py --config configs/radar/radar_features_v1.yaml
```

Create windows:

```powershell
python scripts/radar/make_radar_windows.py --config configs/radar/radar_features_v1.yaml
```

Extract features:

```powershell
python scripts/radar/extract_radar_features.py --config configs/radar/radar_features_v1.yaml
```

Run rule baseline:

```powershell
python scripts/radar/run_radar_rules.py --config configs/radar/radar_rules_v1.yaml
```

Evaluate rule or model output:

```powershell
python scripts/radar/eval_radar_baseline.py --config configs/radar/radar_rules_v1.yaml
```

Train the reserved Random Forest baseline only when radar labels are ready:

```powershell
python scripts/radar/train_radar_baseline.py --config configs/radar/radar_rf_v1.yaml
```

Export board C rule constants:

```powershell
python scripts/radar/export_radar_rules_c.py --config configs/radar/radar_rules_v1.yaml
```

## Fusion Interface

Fusion should consume:

- MIC cough probability
- `radar_state`
- `radar_quality_score`
- `trust_factor`

The first fusion version should be a rule state machine configured by
`configs/fusion/fusion_rules_v1.yaml`. A small MLP is deferred until the rule
baseline and person-grouped evaluation are stable.

## Board Interface

`scripts/radar/export_radar_rules_c.py` exports:

- `deploy/radar/radar_rules_v1.h`
- `deploy/radar/radar_rules_v1.c`

These generated files contain thresholds and state enum values for board-side
integration. Treat them as generated deployment artifacts; regenerate from
`configs/radar/radar_rules_v1.yaml` whenever thresholds change.
