# Audio Baseline V2 Label Changes

## Inputs

- Requested labels: `data/labels/audio_labels.csv`
- Effective base labels: `data/labels/formal_audio_50/audio_labels.csv`
- Review sheet: `review/audio_baseline_v1/review_sheet.csv`
- Output labels: `data/labels/audio_labels_v2.csv`

## Source Note

Requested input `data/labels/audio_labels.csv` did not contain reviewed source_audio rows, so v2 was generated from `data/labels/formal_audio_50/audio_labels.csv`.

Original label files and WAV files were not modified.

## Summary

- Base rows: 50
- V2 rows: 58
- Reviewed intervals: 5
- Modified intervals written: 5
- Excluded intervals: 0
- Hard negative intervals: 2
- V2 label counts: {'non_cough': 32, 'cough': 26}

## Decision Counts

| manual_decision | count |
| --- | ---: |
| hard_negative | 2 |
| relabel_to_non_cough | 3 |

## Changed / Reviewed Intervals

| audio_file | start_time | end_time | original_label | new_label | manual_decision | hard_negative | change_reason |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `data/formal_audio_50/cough_0024.wav` | 1.000 | 2.000 | cough | non_cough | relabel_to_non_cough | False | manual_review_relabel_to_non_cough source=data/formal_audio_50/cough_0024.wav interval=1.000-2.000s label=cough->non_cough cough_prob=0.390711 |
| `data/formal_audio_50/cough_0024.wav` | 2.000 | 3.000 | cough | non_cough | relabel_to_non_cough | False | manual_review_relabel_to_non_cough source=data/formal_audio_50/cough_0024.wav interval=2.000-3.000s label=cough->non_cough cough_prob=0.392354 |
| `data/formal_audio_50/non_cough_0023.wav` | 0.000 | 2.500 | non_cough | non_cough | hard_negative | True | manual_review_hard_negative source=data/formal_audio_50/non_cough_0023.wav interval=0.000-2.500s cough_prob=0.830644 |
| `data/formal_audio_50/non_cough_0023.wav` | 14.000 | 16.500 | non_cough | non_cough | hard_negative | True | manual_review_hard_negative source=data/formal_audio_50/non_cough_0023.wav interval=14.000-16.500s cough_prob=0.720138 |
| `data/formal_audio_50/non_cough_0023.wav` | 20.000 | 22.000 | non_cough | non_cough | relabel_to_non_cough | False | manual_review_relabel_to_non_cough source=data/formal_audio_50/non_cough_0023.wav interval=20.000-22.000s label=non_cough->non_cough cough_prob=0.987624 |

## Notes For Next Training

- Use `data/labels/audio_labels_v2.csv` only after updating the training config paths if this v2 file is intended for the formal 50-person dataset.
- `hard_negative=True` rows are still labeled `non_cough`; training code can optionally use this column later for sampling or loss weighting.
- The two `cough_0024.wav` reviewed intervals were relabeled to `non_cough`, so the original full-audio cough label was split into cough/non_cough/cough segments.
