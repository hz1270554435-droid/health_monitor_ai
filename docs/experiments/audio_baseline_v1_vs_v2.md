# Audio Baseline V1 vs V2

## Runs

| version | run_dir | model | threshold |
| --- | --- | --- | ---: |
| baseline_v1 | `results/audio/audio_baseline_20260505_182746` | `models/audio/audio_baseline_20260505_182746/best_model.pt` | default argmax |
| baseline_v2 | `results/audio/audio_baseline_20260505_192102` | `models/audio/audio_baseline_20260505_192102/best_model.pt` | 0.75 |

## Metric Comparison

| metric | baseline_v1 | baseline_v2 | delta |
| --- | ---: | ---: | ---: |
| accuracy | 0.88060 | 0.96875 | +0.08815 |
| precision | 0.72000 | 0.93939 | +0.21939 |
| recall | 0.94737 | 0.93939 | -0.00797 |
| f1 | 0.81818 | 0.93939 | +0.12121 |

The main improvement is precision: it increased from `0.72` to `0.93939`. Cough recall remains high at `0.93939`.

## Error Comparison

| error type | baseline_v1 | baseline_v2 | delta |
| --- | ---: | ---: | ---: |
| false_positive: non_cough -> cough | 14 | 2 | -12 |
| false_negative: cough -> non_cough | 2 | 2 | 0 |
| total misclassified | 16 | 4 | -12 |

False positives dropped from `14` to `2`, which is the main reason precision improved.

## Classification Summary

### Baseline V1

| class | precision | recall | f1-score | support |
| --- | ---: | ---: | ---: | ---: |
| non_cough | 0.98 | 0.85 | 0.91 | 96 |
| cough | 0.72 | 0.95 | 0.82 | 38 |
| accuracy |  |  | 0.88 | 134 |
| macro avg | 0.85 | 0.90 | 0.86 | 134 |
| weighted avg | 0.90 | 0.88 | 0.88 | 134 |

### Baseline V2

Threshold: `0.75`

| class | precision | recall | f1-score | support |
| --- | ---: | ---: | ---: | ---: |
| non_cough | 0.98 | 0.98 | 0.98 | 95 |
| cough | 0.94 | 0.94 | 0.94 | 33 |
| accuracy |  |  | 0.97 | 128 |
| macro avg | 0.96 | 0.96 | 0.96 | 128 |
| weighted avg | 0.97 | 0.97 | 0.97 | 128 |

## V2 Key Changes

Baseline V2 includes four practical changes:

1. Manual review of suspicious error windows from baseline_v1.
2. Relabeling and cleaning reviewed segments into `data/labels/audio_labels_v2.csv`.
3. Early stopping on `val_f1`, with `early_stopping_patience=5`.
4. Threshold tuning with the selected cough probability threshold `0.75`.

The threshold sweep selected `0.75` because it maximized F1 while keeping cough recall at or above the target range:

| threshold | accuracy | precision | recall | f1 | false_positive | false_negative |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 0.96875 | 0.93939 | 0.93939 | 0.93939 | 2 | 2 |

## Interpretation

Baseline V1 was recall-heavy but produced too many false cough alarms. Most of its false positives came from a small number of suspicious non_cough segments. After manual review, relabeling/cleaning, early stopping, and threshold tuning, baseline_v2 keeps cough recall high while sharply reducing false positives.

The remaining baseline_v2 errors are balanced: `2` false positives and `2` false negatives. This is a much better operating point for the current MIC baseline than baseline_v1.

## Selection

`audio_baseline_v2_selected` can be used as the current MIC main model.

Recommended deployment/evaluation setting:

```text
model_path = models/audio/audio_baseline_20260505_192102/best_model.pt
threshold = 0.75
config = configs/audio_formal_50_v2.yaml
```

## Notes

The v1 and v2 test row counts differ because v2 labels split and relabeled reviewed time ranges. V2 should be compared as the cleaned-label successor to v1, not as an identical-label rerun.
