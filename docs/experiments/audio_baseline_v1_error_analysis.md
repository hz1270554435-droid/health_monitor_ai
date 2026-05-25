# Audio Baseline V1 Error Analysis

Run directory: `results/audio/audio_baseline_20260505_182746`

Model checkpoint: `models/audio/audio_baseline_20260505_182746/best_model.pt`

## Summary

Evaluation split: `test`

Test windows: 134

Test metrics:

| metric | value |
| --- | ---: |
| accuracy | 0.8806 |
| precision | 0.7200 |
| recall | 0.9474 |
| f1 | 0.8182 |

Training summary from `train_log.csv`:

| item | value |
| --- | ---: |
| epochs run | 30 |
| best epoch by val_f1 | 7 |
| best val_f1 | 0.9649 |
| best val_precision | 1.0000 |
| best val_recall | 0.9322 |
| final epoch val_f1 | 0.8762 |

The model is tuned toward cough recall. On the test set it misses only 2 cough windows, but it produces 14 non_cough false positives.

## Classification Report

| class | precision | recall | f1-score | support |
| --- | ---: | ---: | ---: | ---: |
| non_cough | 0.98 | 0.85 | 0.91 | 96 |
| cough | 0.72 | 0.95 | 0.82 | 38 |
| accuracy |  |  | 0.88 | 134 |
| macro avg | 0.85 | 0.90 | 0.86 | 134 |
| weighted avg | 0.90 | 0.88 | 0.88 | 134 |

## Error Counts

| error type | count |
| --- | ---: |
| false positive: non_cough -> cough | 14 |
| false negative: cough -> non_cough | 2 |
| total misclassified | 16 |

## True / Predicted Label Counts

Rows shown here are only misclassified samples.

| true_label | pred_label | count |
| --- | --- | ---: |
| cough | non_cough | 2 |
| non_cough | cough | 14 |

## Errors By Person

| person_id | true_label | pred_label | count |
| --- | --- | --- | ---: |
| P_FORMAL_048 | non_cough | cough | 13 |
| P_FORMAL_024 | cough | non_cough | 2 |
| P_FORMAL_049 | non_cough | cough | 1 |

Most errors are concentrated in one non_cough test file:

| person_id | label | audio_file | test_windows | errors | error_rate |
| --- | --- | --- | ---: | ---: | ---: |
| P_FORMAL_023 | cough | `data/formal_audio_50/cough_0023.wav` | 11 | 0 | 0.0000 |
| P_FORMAL_024 | cough | `data/formal_audio_50/cough_0024.wav` | 9 | 2 | 0.2222 |
| P_FORMAL_025 | cough | `data/formal_audio_50/cough_0025.wav` | 18 | 0 | 0.0000 |
| P_FORMAL_048 | non_cough | `data/formal_audio_50/non_cough_0023.wav` | 43 | 13 | 0.3023 |
| P_FORMAL_049 | non_cough | `data/formal_audio_50/non_cough_0024.wav` | 11 | 1 | 0.0909 |
| P_FORMAL_050 | non_cough | `data/formal_audio_50/non_cough_0025.wav` | 42 | 0 | 0.0000 |

## Errors By Scene

`split_manifest.csv` does not contain a `scene` column, so scene-level error statistics were not available for this run.

## Top 10 Non-Cough False Positives By Cough Probability

These are non_cough windows predicted as cough with the highest cough probability.

| rank | sample_id | audio_file | person_id | cough_prob | start_time | end_time |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `clip_non_cough_0023_0042` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.9876 | 21.0 | 22.0 |
| 2 | `clip_non_cough_0023_0041` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.9847 | 20.5 | 21.5 |
| 3 | `clip_non_cough_0023_0000` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.8306 | 0.0 | 1.0 |
| 4 | `clip_non_cough_0023_0040` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.8292 | 20.0 | 21.0 |
| 5 | `clip_non_cough_0023_0003` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.8196 | 1.5 | 2.5 |
| 6 | `clip_non_cough_0023_0002` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.7335 | 1.0 | 2.0 |
| 7 | `clip_non_cough_0023_0031` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.7201 | 15.5 | 16.5 |
| 8 | `clip_non_cough_0023_0029` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.6825 | 14.5 | 15.5 |
| 9 | `clip_non_cough_0023_0030` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.6720 | 15.0 | 16.0 |
| 10 | `clip_non_cough_0023_0028` | `data/formal_audio_50/non_cough_0023.wav` | P_FORMAL_048 | 0.6496 | 14.0 | 15.0 |

Observation: the strongest false positives are clustered in `non_cough_0023.wav`, especially near `20.0s-22.0s` and `0.0s-2.5s`. This file should be manually reviewed first; it may contain cough-like transients, speech bursts, throat clearing, impacts, or label noise.

## Cough False Negatives With Lowest Cough Probability

These are cough windows predicted as non_cough, sorted by lowest cough probability.

| sample_id | audio_file | person_id | cough_prob | start_time | end_time |
| --- | --- | --- | ---: | ---: | ---: |
| `clip_cough_0024_0002` | `data/formal_audio_50/cough_0024.wav` | P_FORMAL_024 | 0.3907 | 1.0 | 2.0 |
| `clip_cough_0024_0004` | `data/formal_audio_50/cough_0024.wav` | P_FORMAL_024 | 0.3924 | 2.0 | 3.0 |

Observation: both missed cough windows are from the same cough file, `cough_0024.wav`. The model is not broadly missing cough; it is weak on this specific sample or these local windows.

## Next Data To Add

Prioritize the next data collection / curation round in this order:

1. Add more hard non_cough negatives similar to `non_cough_0023.wav`, especially cough-like non-cough events. This is the dominant error source: 13 of 14 false positives come from one file.
2. Manually inspect `non_cough_0023.wav` at `0.0s-2.5s`, `14.0s-16.5s`, and `20.0s-22.0s`. If there is actual cough or cough-like throat clearing, either relabel those intervals or exclude them from clean negative training.
3. Add negative examples for short impulsive sounds, speech bursts, throat clearing, laugh-like sounds, microphone handling, clicks, and impact noises. These are common sources of false cough activation.
4. Add more cough samples like `cough_0024.wav`, especially lower-energy or partial cough windows around `1.0s-3.0s`. These are the only false negatives.
5. Increase test-set diversity. This test split has only 6 people and 6 source files, so one difficult non_cough file strongly affects precision.
6. Add a `scene` column to labels or manifest if the source data can provide it, for example `quiet`, `speech`, `music`, `movement`, `background_noise`, `device_handling`, or `unknown`. That will make future error analysis more actionable.
7. Keep person-level grouping. Do not split by clip/window, because that would leak subject or file-specific acoustics across train/val/test.

## Training Notes

The best validation epoch was epoch 7, while training continued to epoch 30. Later epochs show validation F1 instability, for example epoch 25 had `val_f1=0.5930`. The saved checkpoint is still selected by best validation F1, but the next training iteration should consider enabling early stopping using the configured `early_stopping_patience`.
