# PSoC E84 Audio Deploy Checklist

## Current Candidate Models

| model | path | status |
| --- | --- | --- |
| float ONNX | `models/audio/export/audio_baseline_v2.onnx` | exported and ONNXRuntime verified |
| int8 ONNX QDQ | `models/audio/export/audio_baseline_v2_int8.onnx` | static QDQ quantized and test-evaluated |

Input shape:

```text
[1, 1, 40, 101]
```

Label map:

```text
0 = non_cough
1 = cough
```

Decision threshold:

```text
cough_prob >= 0.75 -> cough
cough_prob < 0.75  -> non_cough
```

## Metrics

### Float ONNX

| metric | value |
| --- | ---: |
| accuracy | 0.96875 |
| precision | 0.93939 |
| recall | 0.93939 |
| f1 | 0.93939 |
| false_positive | 2 |
| false_negative | 2 |

ONNX export consistency:

| check | value |
| --- | ---: |
| max_abs_diff | 1.907e-6 |
| mean_abs_diff | 1.431e-6 |
| opset | 17 |

### Int8 ONNX QDQ

| metric | value |
| --- | ---: |
| accuracy | 0.96094 |
| precision | 0.91176 |
| recall | 0.93939 |
| f1 | 0.92537 |
| false_positive | 3 |
| false_negative | 2 |

Quantization summary:

| item | value |
| --- | --- |
| format | ONNX static int8 QDQ |
| representative data | train + val Log-Mel features |
| representative rows | 740 |
| test rows | 128 |
| f1 drop vs float | 0.01402 |

## Current Limitations

- TFLite int8 has not been generated yet.
- The current quantized model is ONNX QDQ int8.
- ONNX QDQ graph input and output are still `float32`; quantization occurs inside the graph.
- `input_scale`, `input_zero_point`, `output_scale`, and `output_zero_point` are not available for the current ONNX QDQ artifact.
- Board-side preprocessing must match the PC Log-Mel pipeline exactly before model accuracy can be trusted.

## Toolchain Checks Needed

Verify these items in ModusToolbox / Infineon AI tools before committing the deployment path:

| check | result |
| --- | --- |
| Can the tool import `models/audio/export/audio_baseline_v2.onnx`? | TBD |
| Can the tool import ONNX opset 17? | TBD |
| Can the tool import QDQ int8 ONNX? | TBD |
| Are `QuantizeLinear` / `DequantizeLinear` nodes supported? | TBD |
| Does the generated target model preserve input shape `[1, 1, 40, 101]`? | TBD |
| Does the tool require NHWC instead of NCHW? | TBD |
| If ONNX QDQ is unsupported, is TFLite int8 required? | TBD |

If ONNX QDQ int8 is not supported, convert the selected model to TFLite int8 and re-run quantized evaluation before deployment.

## Board-Side Verification Order

1. Fixed Log-Mel feature test vector
   - Use `deploy/test_vectors/audio/`.
   - Feed `input_feature_float32.bin` or the arrays in `audio_test_vectors.h`.
   - Compare logits and thresholded labels against `expected_output.json`.

2. Fixed PCM buffer + board-side Log-Mel
   - Feed a fixed 16 kHz PCM buffer.
   - Run board-side Log-Mel extraction.
   - Compare the produced `[1, 1, 40, 101]` feature against the PC feature.
   - Then compare model logits and cough decision.

3. Real-time PDM microphone
   - Capture PDM audio and convert to 16 kHz PCM.
   - Run 1.0 s window and 0.5 s hop.
   - Run board-side Log-Mel 40-bin preprocessing.
   - Run model inference.
   - Apply `cough_prob >= 0.75`.

## Preprocessing Requirements

| item | value |
| --- | ---: |
| sample_rate | 16000 Hz |
| window | 1.0 s |
| hop | 0.5 s |
| n_mels | 40 |
| expected feature shape | `[1, 1, 40, 101]` |

The board-side normalization must be identical to the PC preprocessing. Differences in Mel filterbank, log scaling, padding, or normalization can cause deployment mismatch even when the model import succeeds.

## Deployment Artifacts

```text
models/audio/export/audio_baseline_v2.onnx
models/audio/export/audio_baseline_v2_int8.onnx
models/audio/export/export_report.json
models/audio/export/quantization_report.json
models/audio/selected/selected_threshold.json
models/audio/selected/label_map.json
deploy/test_vectors/audio/
docs/deployment/audio_model_deployment_spec.md
```
