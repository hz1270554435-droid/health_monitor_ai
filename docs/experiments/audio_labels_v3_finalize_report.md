# audio_labels_v3 Finalize Report

## Inputs

- candidate: `D:\cough_model_train\health_monitor_ai\data\labels\audio_labels_v3_candidate.csv`
- output labels: `D:\cough_model_train\health_monitor_ai\data\labels\audio_labels_v3.csv`

## Cleanup Rules Applied

- sample_weight filled with 1.0: `58`
- hard_negative_source filled with legacy_hard_negative: `2`
- protected columns unchanged: `True`
- candidate rows: `2072`
- output rows: `2072`

## sample_weight Fix Examples

| clip_id | audio_file | start_time | end_time | label | split | hard_negative_source |
| --- | --- | --- | --- | --- | --- | --- |
| clip_cough_0001 | data/formal_audio_50/cough_0001.wav | 0.0 | 9.78 | cough | train |  |
| clip_cough_0002 | data/formal_audio_50/cough_0002.wav | 0.0 | 4.5 | cough | train |  |
| clip_cough_0003 | data/formal_audio_50/cough_0003.wav | 0.0 | 9.84 | cough | train |  |
| clip_cough_0004 | data/formal_audio_50/cough_0004.wav | 0.0 | 7.26 | cough | train |  |
| clip_cough_0005 | data/formal_audio_50/cough_0005.wav | 0.0 | 9.78 | cough | train |  |
| clip_cough_0006 | data/formal_audio_50/cough_0006.wav | 0.0 | 9.84 | cough | train |  |
| clip_cough_0007 | data/formal_audio_50/cough_0007.wav | 0.0 | 9.84 | cough | train |  |
| clip_cough_0008 | data/formal_audio_50/cough_0008.wav | 0.0 | 7.5 | cough | train |  |
| clip_cough_0009 | data/formal_audio_50/cough_0009.wav | 0.0 | 9.9 | cough | train |  |
| clip_cough_0010 | data/formal_audio_50/cough_0010.wav | 0.0 | 1.44 | cough | train |  |

## hard_negative_source Fix Examples

| clip_id | audio_file | start_time | end_time | label | split | hard_negative_source |
| --- | --- | --- | --- | --- | --- | --- |
| clip_non_cough_0023_v2_01 | data/formal_audio_50/non_cough_0023.wav | 0.0 | 2.5 | non_cough | test |  |
| clip_non_cough_0023_v2_03 | data/formal_audio_50/non_cough_0023.wav | 14.0 | 16.5 | non_cough | test |  |

## Dataset Check

- status: `passed`
- audio missing: `0`
- duplicate audio segment: `0`
- person_id split leakage: `0`
- sample_weight missing_rows: `0`
- sample_weight non_positive_rows: `0`
- hard_negative empty source rows: `0`
- label_counts: `{'cough': 26, 'non_cough': 2046}`
- split_counts: `{'test': 14, 'train': 2050, 'val': 8}`
- hard_negative_source_counts: `{'': 56, 'boundary_clean_non_cough': 296, 'clean_non_cough': 92, 'human_cough_like': 96, 'legacy_hard_negative': 2, 'model_false_alarm': 1530}`

## Errors

- none
