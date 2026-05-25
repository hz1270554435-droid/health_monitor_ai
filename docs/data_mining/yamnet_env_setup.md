# YAMNet hard-mining environment

Use a separate virtual environment so TensorFlow dependencies do not alter the
main PyTorch training environment.

```powershell
py -3.11 -m venv .venv_yamnet
.\.venv_yamnet\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_yamnet.txt
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

The scan script still loads the selected PyTorch baseline checkpoint to compute
`baseline_cough_prob`, so the isolated YAMNet environment needs both
TensorFlow/YAMNet dependencies and a CPU PyTorch install.

YAMNet is used only for pre-screening. It does not modify formal labels and it
does not train the MIC model.

## Smoke flow

Disk space is limited. Do not pre-copy the external non-cough pool. Scan the
original folder first, then export only the selected review/hard clips.

Scan only the first three WAV files from the original source:

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

Expand to 100 files:

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

Export a small review batch:

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

After manual review, generate a new label file without overwriting v2:

```powershell
python scripts/build_labels_from_review.py `
  --base-labels data/labels/audio_labels_v2.csv `
  --review-sheet review/yamnet_hard_mining_v1/review_sheet.csv `
  --out-labels data/labels/audio_labels_v3.csv `
  --out-report docs/experiments/audio_labels_v3_from_yamnet_review.md
```

Leave pseudo labels disabled by default. Add `--include-pseudo` only after spot
checking the candidate quality.
