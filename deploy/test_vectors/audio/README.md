# Audio Deploy Test Vectors

Input shape: `[1, 1, 40, 101]`

Each sample directory contains:

- `input_feature_float32.npy`: NumPy float32 Log-Mel tensor with shape `[1, 1, 40, 101]`
- `input_feature_float32.bin`: raw little-endian float32 tensor data in NCHW order
- `expected_output.json`: PyTorch and ONNX logits plus thresholded expected label

Selection:

- 5 correct cough samples
- 5 correct non_cough samples
- 4 remaining v2 misclassified samples

Threshold: `0.75`

The C header `audio_test_vectors.h` contains two full float32 input arrays for minimal board-side inference tests.

Total generated samples: 14
