# Audio Model Deployment Spec

## Model

ONNX model:

```text
models/audio/export/audio_baseline_v2.onnx
```

Selected checkpoint source:

```text
models/audio/selected/audio_baseline_v2_best_model.pt
```

## Input

Input shape:

```text
[1, 1, 40, 101]
```

Input dimensions:

| dimension | value | meaning |
| --- | ---: | --- |
| batch | 1 | one feature window |
| channel | 1 | single Log-Mel feature channel |
| n_mels | 40 | 40 Mel frequency bins |
| time_frames | 101 | feature frames for one 1 s window |

Input dtype for the exported float ONNX model is `float32`.

## Output

The model outputs logits with shape:

```text
[1, 2]
```

Class order:

```text
0 = non_cough
1 = cough
```

Apply softmax to logits, then read class index `1` as `cough_prob`.

Decision threshold:

```text
cough_prob >= 0.75 -> cough
cough_prob < 0.75  -> non_cough
```

## Preprocessing

Board-side preprocessing must match the PC preprocessing.

Required audio settings:

| item | value |
| --- | ---: |
| sample_rate | 16000 Hz |
| window | 1.0 s |
| hop | 0.5 s |
| feature | Log-Mel |
| n_mels | 40 |

The board-side normalization must be identical to the PC pipeline. Any difference in log scale, Mel filterbank, padding, clipping, or normalization can shift the model output.

## ONNX Export Consistency

PyTorch and ONNXRuntime outputs were compared with a dummy input matching the real feature shape.

| check | value |
| --- | ---: |
| max_abs_diff | 1.907e-6 |
| mean_abs_diff | 1.431e-6 |
| ONNX opset | 17 |
| input shape | `[1, 1, 40, 101]` |
| output shape | `[1, 2]` |

The exported ONNX model is numerically consistent with the PyTorch checkpoint for deployment candidate testing.

## Deployment Steps

1. Run int8 quantization.
2. Validate generated test vectors on PC and target.
3. Run board-side fixed-feature inference using `deploy/test_vectors/audio/`.
4. Integrate board-side real-time PDM audio capture.
5. Run board-side real-time preprocessing and model inference with the same 1 s window and 0.5 s hop.
6. Apply the selected threshold `0.75` to `cough_prob`.

## Related Artifacts

```text
models/audio/export/export_report.json
models/audio/export/audio_baseline_v2_int8.onnx
models/audio/export/quantization_report.json
deploy/test_vectors/audio/
models/audio/selected/selected_threshold.json
models/audio/selected/label_map.json
```
