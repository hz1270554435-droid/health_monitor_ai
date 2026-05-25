# audio v3.1 data mining pipeline

This document tracks the v3.1 data mining flow. The flow adds new reviewed
candidate labels beside `audio_labels_v3.csv`; it does not overwrite v3 labels,
v3 model artifacts, or start OPERA distillation.

## Inputs

- Config: `configs/audio_mining_v3_1.yaml`
- Base labels: `data/labels/audio_labels_v3.csv`
- v3 checkpoint: `models/audio/audio_baseline_v3_board_htk_hardneg/best_model.pt`
- External sources:
  - `D:/cough_model_train/DATA/board_live_false_alarm`
  - `D:/cough_model_train/DATA/board_live_cough`
  - `D:/cough_model_train/DATA/board_live_clean_non_cough`
  - `D:/cough_model_train/DATA/cough/cough_READY/cough`

## Board-live capture

Build the audio-only CSV_EXPORT firmware from the top-level workspace:

```powershell
. .\tools\enter_mtb_env.ps1
make firmware-csv-audio
```

After flashing the built firmware, capture one source domain at a time into the
matching external root. For board-live false alarms:

```powershell
python firmware/tools/capture_csv.py `
  --port auto `
  --baud 2000000 `
  --duration 300 `
  --output-dir D:/cough_model_train/DATA/board_live_false_alarm `
  --session-output-dir . `
  --person-id board_live_p01 `
  --scene board_false_alarm `
  --audio-channels 1 `
  --audio-mix-mode select-best
```

Use the same command with `board_live_cough` or
`board_live_clean_non_cough` as `--output-dir` and `--scene` when recording
intentional cough or clean non-cough sessions. The capture tool stores WAV/JSON
sessions under the timestamped run directory; the v3.1 manifest builder scans
the external roots recursively. `--audio-mix-mode select-best` mirrors the
board frontend's block-energy channel selector for mono WAV export, and the
session JSON records firmware `capture_mode` metadata from the serial stream.

## Commands

```powershell
python scripts/build_audio_mining_manifest.py --config configs/audio_mining_v3_1.yaml
python scripts/scan_v3_board_model_audio.py --config configs/audio_mining_v3_1.yaml --overwrite
python scripts/scan_yamnet_v3_1_audio.py --config configs/audio_mining_v3_1.yaml --overwrite
python scripts/create_v3_1_review_plan.py --config configs/audio_mining_v3_1.yaml --overwrite
python scripts/export_v3_1_review_clips.py --config configs/audio_mining_v3_1.yaml --group-by-bucket --overwrite
```

After manual review:

```powershell
python scripts/build_v3_1_candidate_from_manual_reviews.py --config configs/audio_mining_v3_1.yaml --overwrite
python scripts/check_dataset.py --config configs/audio_baseline_v3_1_candidate.yaml --out results/check_audio_labels_v3_1_candidate/report.json
python scripts/preprocess_audio.py --config configs/audio_baseline_v3_1_candidate.yaml --output-dir results/preprocess_smoke_v3_1_candidate --limit 12 --overwrite
python scripts/train_audio_task.py --config configs/audio_baseline_v3_1_candidate.yaml --manifest results/preprocess_smoke_v3_1_candidate/audio_manifest.csv --run-name one_epoch_smoke_v3_1_candidate --epochs 1
```

## Manual decisions

`manual_decision` must be one of `cough`, `hard_negative`,
`clean_non_cough`, `uncertain`, or `exclude`. OPERA fields are reserved in the
CSV interfaces but are intentionally not populated or used in v3.1 mining.
