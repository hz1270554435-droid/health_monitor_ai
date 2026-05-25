# audio_labels_v3_candidate from YAMNet manual review

This is a candidate label build only. It does not train a model and does not overwrite audio_labels_v2.csv.

## Summary

- base_labels: `D:\cough_model_train\health_monitor_ai\data\labels\audio_labels_v2.csv`
- out_labels: `D:\cough_model_train\health_monitor_ai\data\labels\audio_labels_v3_candidate.csv`
- out_merged: `D:\cough_model_train\health_monitor_ai\review\yamnet_full_non_cough_merged\manual_review_merged.csv`
- base_rows: 58
- merged_rows: 5334
- reviewed_rows: 2050
- added_rows: 2014
- skipped_rows: 36
- manual_conflict_rows: 0
- duplicate_skipped_rows: 0
- invalid_manual_decision_count: 0

## Label Counts In Added Rows

- non_cough: 2014

## Hard Negative Source Counts

- model_false_alarm: 1530
- boundary_clean_non_cough: 296
- human_cough_like: 96
- clean_non_cough: 92

## Derived Review Class Counts

- unreviewed: 3284
- model_false_alarm_hard_negative: 1530
- boundary_clean_non_cough: 296
- human_cough_like_hard_negative: 96
- clean_non_cough: 92
- exclude_low_quality: 36

## Manual Decision Counts

- <blank>: 3284
- clean_non_cough: 1918
- hard_negative: 96
- exclude: 36

## Source Audio Stats

- added_source_audio_count: 1268
- max_added_rows_per_source_audio: 6

## Skipped Rows By Derived Class

- exclude_low_quality: 36
