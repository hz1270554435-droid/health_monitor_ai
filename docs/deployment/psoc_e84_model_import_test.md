# PSoC E84 Model Import Test Record

Source checklist:

```text
docs/deployment/psoc_e84_audio_deploy_checklist.md
```

## Candidate Models

| format | path | status |
| --- | --- | --- |
| float ONNX | `models/audio/export/audio_baseline_v2.onnx` | pending tool import test |
| int8 ONNX QDQ | `models/audio/export/audio_baseline_v2_int8.onnx` | pending tool import test |
| TFLite int8 | `models/audio/export/audio_baseline_v2_int8.tflite` | not generated yet |

Expected input shape:

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
```

## Float ONNX Import Test

| item | result | notes |
| --- | --- | --- |
| Tool / version | TBD | ModusToolbox / Infineon AI tool version |
| Import succeeded | TBD | yes / no |
| ONNX opset 17 supported | TBD | model opset is 17 |
| Input shape preserved as `[1, 1, 40, 101]` | TBD | verify generated model input |
| Output shape preserved as `[1, 2]` | TBD | logits for non_cough/cough |
| Tool requires NHWC | TBD | yes / no |
| If NHWC required | TBD | re-export as NHWC or add transpose wrapper |
| Fixed feature test vector passed | TBD | compare against `deploy/test_vectors/audio/` |
| Selected for deployment | TBD | yes / no |

## Int8 ONNX QDQ Import Test

| item | result | notes |
| --- | --- | --- |
| Tool / version | TBD | ModusToolbox / Infineon AI tool version |
| Import succeeded | TBD | yes / no |
| ONNX opset 17 supported | TBD | model opset is 17 |
| QDQ int8 supported | TBD | yes / no |
| `QuantizeLinear` supported | TBD | required by QDQ graph |
| `DequantizeLinear` supported | TBD | required by QDQ graph |
| Input shape preserved as `[1, 1, 40, 101]` | TBD | QDQ model graph input is float32 |
| Output shape preserved as `[1, 2]` | TBD | QDQ model graph output is float32 logits |
| Tool requires NHWC | TBD | yes / no |
| If NHWC required | TBD | re-export as NHWC or add transpose wrapper |
| Fixed feature test vector passed | TBD | compare against `deploy/test_vectors/audio/` |
| Selected for deployment | TBD | yes / no |

## ONNX Opset 17 Support

| check | result | notes |
| --- | --- | --- |
| Float ONNX opset 17 import | TBD | `audio_baseline_v2.onnx` |
| Int8 ONNX QDQ opset 17 import | TBD | `audio_baseline_v2_int8.onnx` |
| Unsupported operators, if any | TBD | list operator names |

## QDQ Operator Support

| operator | support | notes |
| --- | --- | --- |
| `QuantizeLinear` | TBD | required for int8 ONNX QDQ |
| `DequantizeLinear` | TBD | required for int8 ONNX QDQ |

If either operator is unsupported, do not use the current ONNX QDQ artifact directly. Use float ONNX or generate a tool-supported TFLite int8 / native format instead.

## Layout Check

Current exported model layout:

```text
NCHW = [batch, channel, n_mels, time_frames] = [1, 1, 40, 101]
```

| check | result | notes |
| --- | --- | --- |
| Tool accepts NCHW directly | TBD | yes / no |
| Tool requires NHWC | TBD | yes / no |
| Required NHWC shape, if applicable | TBD | likely `[1, 40, 101, 1]` |
| Re-export needed | TBD | yes / no |
| Transpose wrapper needed | TBD | yes / no |

If the tool requires NHWC, record the chosen mitigation:

```text
TBD: re-export model with NHWC input / add transpose wrapper / convert preprocessing output to expected layout
```

## Final Deployment Format Decision

Select one after import and fixed-vector validation:

| option | selected | reason |
| --- | --- | --- |
| float ONNX | TBD | TBD |
| int8 ONNX QDQ | TBD | TBD |
| TFLite int8 | TBD | TBD |
| other | TBD | TBD |

Final selected format:

```text
TBD
```

## Validation Order

1. Import candidate model into ModusToolbox / Infineon AI tool.
2. Confirm model input shape and layout.
3. Run fixed Log-Mel feature vectors from `deploy/test_vectors/audio/`.
4. Compare board/tool output against `expected_output.json`.
5. If fixed feature vectors pass, test fixed PCM buffer plus board-side Log-Mel.
6. If PCM path passes, move to real-time PDM microphone validation.

## Notes

- Do not modify the model artifacts during this import test.
- Record any conversion command, tool warning, unsupported operator, or shape/layout change in this file.
- If the final deployment path uses a converted model, create a new export report for that artifact before firmware integration.
