# Health Monitor AI

Edge-AI respiratory health monitoring prototype using microphone audio and LD6002 radar. The first implemented stage is the MIC cough/non_cough baseline. Radar and fusion configs are scaffolded for later stages.

## Project Layout

```text
configs/                 YAML configs for reproducible runs
data/labels/             sessions.csv and audio_labels.csv
data/raw/audio/          local WAV files, never committed
data/processed/          generated Log-Mel features, ignored by Git
scripts/                 command-line entry points
src/audio/               audio feature, dataset, and model code
src/common/              shared config, split, IO, and metric helpers
models/audio/            trained checkpoints, ignored by Git
results/audio/           metrics and plots, ignored by Git
```

## Data Format

Create `data/labels/audio_labels.csv` with these columns:

```csv
clip_id,session_id,person_id,start_time,end_time,label,audio_file
```

Labels must be `cough` or `non_cough`. Splits are grouped by `person_id`; do not split randomly by `clip_id`.

## Windows + Python 3.11 CPU Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch CPU wheels first. This avoids CUDA/GPU dependencies and includes `torchaudio` for the MIC baseline:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Then install the regular Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

No public dataset is downloaded by this repository.

## MIC Baseline Workflow

Validate labels and WAV references:

```powershell
python scripts/check_dataset.py --config configs/audio_baseline.yaml
```

Explicit dataset report command:

```powershell
python scripts/check_dataset.py --labels data/labels/audio_labels.csv --audio-root data/raw/audio --out results/dataset_report.json
```

Extract 16 kHz, 1.0 s window, 0.5 s hop, 40-bin Log-Mel features:

```powershell
python scripts/preprocess_audio.py --config configs/audio_baseline.yaml --overwrite
```

Train the small CNN baseline:

```powershell
python scripts/train_audio_baseline.py --config configs/audio_baseline.yaml
```

For a smoke run on a tiny local sample:

```powershell
python scripts/run_pipeline.py --config configs/audio_baseline.yaml --max-rows 12 --epochs 1 --overwrite
```

## YAMNet Hard Sample Mining

YAMNet is used only as a pre-screening tool for hard negatives, suspected cough
candidates, clean non-cough candidates, and uncertain clips. It does not modify
formal labels and it does not train the MIC baseline.

Use an isolated TensorFlow environment as described in
`docs/data_mining/yamnet_env_setup.md`. Keep the main PyTorch training
environment unchanged.

Do not pre-copy the full external non-cough pool. Disk space is limited, so scan
the original folder first and only materialize selected clips after candidate
screening.

Run a three-file scan smoke test directly on the external source:

```powershell
python scripts/yamnet_scan_audio.py `
  --audio-dir D:/cough_model_train/DATA/non_cough `
  --out-dir data/mining_candidates/yamnet_v1 `
  --baseline-config configs/audio_formal_50_v2.yaml `
  --baseline-model models/audio/selected/audio_baseline_v2_best_model.pt `
  --threshold 0.75 `
  --window-sec 1.0 `
  --hop-sec 0.5 `
  --max-files 3
```

Then expand to 100 files before scanning the full source:

```powershell
python scripts/yamnet_scan_audio.py `
  --audio-dir D:/cough_model_train/DATA/non_cough `
  --out-dir data/mining_candidates/yamnet_non_cough_100files `
  --baseline-config configs/audio_formal_50_v2.yaml `
  --baseline-model models/audio/selected/audio_baseline_v2_best_model.pt `
  --threshold 0.75 `
  --window-sec 1.0 `
  --hop-sec 0.5 `
  --max-files 100
```

Export review clips:

```powershell
python scripts/export_review_clips.py `
  --candidate-csv data/mining_candidates/yamnet_non_cough_100files/candidate_pool.csv `
  --out-dir review/yamnet_hard_mining_v1 `
  --top-hard 20 `
  --top-baseline-suspicious 20 `
  --top-pseudo-cough 20 `
  --top-uncertain 20 `
  --context-sec 0.5
```

Build `audio_labels_v3.csv` only after manual review:

```powershell
python scripts/build_labels_from_review.py `
  --base-labels data/labels/audio_labels_v2.csv `
  --review-sheet review/yamnet_hard_mining_v1/review_sheet.csv `
  --out-labels data/labels/audio_labels_v3.csv `
  --out-report docs/experiments/audio_labels_v3_from_yamnet_review.md
```

## Outputs

Each training run writes:

- `models/audio/<run>/best_model.pt`
- `results/audio/<run>/audio_baseline.yaml`
- `results/audio/<run>/metrics.json`
- `results/audio/<run>/classification_report.txt`
- `results/audio/<run>/confusion_matrix.png`
- `results/audio/<run>/split_manifest.csv`

## Rules

Do not modify or delete `data/raw/`. Do not commit raw audio, processed features, checkpoints, or result files. Every generated artifact must be reproducible from `configs/*.yaml`.
