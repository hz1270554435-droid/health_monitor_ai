# v3.2 Train1 Formal Training Summary

- training result: `pass`
- deploy candidate: `false`
- config: `D:\cough_model_train\health_monitor_ai\configs\audio_baseline_v3_2_recall_fix.yaml`
- label: `D:\cough_model_train\health_monitor_ai\data\labels\audio_labels_v3_2_train_weighted.csv`
- weighted label hash: `4fb074da5f15861d1b9ac463315844e82bb3ba2545ac33374bf9be218740913b`
- epochs run: `20`
- best epoch: `15`
- checkpoint: `D:\cough_model_train\health_monitor_ai\models\audio_v3_2_recall_fix\audio_baseline_v3_2_recall_fix_train1\best_model.pt`

## Sampler

- sample_weight active: `True`
- train sampler: `WeightedRandomSampler`
- observed sampler label distribution: `{'non_cough': 2179, 'cough': 837}`
- observed sampler cough fraction: `0.27752`
- train feature expected cough fraction: `0.274147`

## Metrics

- final train loss: `0.11651865037192717`
- final val loss: `0.4142527846065727`
- best val f1: `0.8785046728971962`
- train metrics: `{'accuracy': 0.9290450928381963, 'precision': 0.6243291592128801, 'recall': 0.9886685552407932, 'f1': 0.7653508771929824}`
- val metrics: `{'accuracy': 0.9121621621621622, 'precision': 0.9791666666666666, 'recall': 0.7966101694915254, 'f1': 0.8785046728971962}`
- test metrics: `{'accuracy': 0.8046875, 'precision': 0.5740740740740741, 'recall': 0.9393939393939394, 'f1': 0.7126436781609197}`
- test confusion matrix: `[[72, 23], [2, 31]]`

## v3.1 FP-fix Reference

`{'available': True, 'source': 'D:\\cough_model_train\\health_monitor_ai\\review\\board_live_data_1\\fpfix_train_report.json', 'status': 'passed', 'best_epoch': 15, 'test_metrics': {'accuracy': 0.8828125, 'precision': 0.7647058823529411, 'recall': 0.7878787878787878, 'f1': 0.7761194029850746}, 'note': 'Reference only; v3.2 train1 is not a deployment candidate.'}`

## Commands

- `D:\e84_health_monitor\ml\.venv\Scripts\python.exe scripts/run_v3_2_train1.py --reuse-existing-output --run-name audio_baseline_v3_2_recall_fix_train1` -> `0`
- `D:\e84_health_monitor\ml\.venv\Scripts\python.exe scripts/preprocess_audio.py --config D:\cough_model_train\health_monitor_ai\configs\audio_baseline_v3_2_recall_fix.yaml --overwrite` -> `0`
- `D:\e84_health_monitor\ml\.venv\Scripts\python.exe scripts/train_audio_baseline.py --config D:\cough_model_train\health_monitor_ai\configs\audio_baseline_v3_2_recall_fix.yaml --run-name audio_baseline_v3_2_recall_fix_train1` -> `0`

## Failed Checks

- none

## Not Executed

- ONNX export
- deployment export
- board package generation
- threshold sweep as final decision
- old label modification
- raw WAV/JSON modification
- ml/data/raw write
