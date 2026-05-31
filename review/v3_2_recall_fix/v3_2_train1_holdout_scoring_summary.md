# v3.2 Train1 Holdout Scoring Summary

- status: `passed`
- model: `ds_cnn`
- checkpoint: `D:\cough_model_train\health_monitor_ai\models\audio_v3_2_recall_fix\audio_baseline_v3_2_recall_fix_train1\best_model.pt`
- scored_rows: `8687`
- joined_v3_1_rows: `8687`
- known_error_regression: `not_completed`
- recommended_next_step: `train2`

## Proxy Metrics

- All scenario-derived metrics in this report are proxy metrics unless explicitly marked manual-review-labeled.
- `scenario_canonical == cough` is treated as cough-session proxy positive, not strict ground truth.

## p01 / p02 / p03 Cough-Session Activation Rate

- p01 @0.50: v3.1 `0.10520939734422881` vs v3.2 `0.35444330949948927`
- p02 @0.50: v3.1 `0.1689497716894977` vs v3.2 `0.3595890410958904`
- p03 @0.50: v3.1 `0.16018306636155608` vs v3.2 `0.43935926773455375`

## long_quiet_eval Event-Level FP/hour

- v3.1 @0.50 event FP/hour: `0.6408031031407047``
- v3.2 @0.50 event FP/hour: `0.0``

## FP Source Ranking @0.50

- `throat_clear` rows `538`, v3.1 FP `47`, v3.2 FP `178`, delta `131`
- `speech` rows `1198`, v3.1 FP `38`, v3.2 FP `135`, delta `97`
- `quiet` rows `3407`, v3.1 FP `42`, v3.2 FP `43`, delta `1`
- `sniff` rows `174`, v3.1 FP `4`, v3.2 FP `40`, delta `36`
- `mixed_non_cough` rows `641`, v3.1 FP `92`, v3.2 FP `30`, delta `-62`

## Recommendation

- speech high-score non_cough windows increased
- throat_clear high-score non_cough windows increased
- sniff high-score non_cough windows increased
- sampler direction: increase weight for speech negative windows or equivalent candidate role
- sampler direction: increase weight for throat_clear negative windows or equivalent candidate role
- sampler direction: increase weight for sniff negative windows or equivalent candidate role
- sampler direction: reduce over-emphasis of positive_recall_fix_candidate weights if proxy activation improved but FP sources worsened

## Not Executed

- training
- ONNX export
- deployment export
- board package generation
- training label modification
- old v3.1/v3 model modification
- raw WAV/JSON modification
- ml/data/raw write

## Known-Error Regression

- This stage does not treat known-error regression as pass unless a ready manifest is available and scored.
- current status: `not_completed`
