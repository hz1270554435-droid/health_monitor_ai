from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize exported MIC audio ONNX model.")
    parser.add_argument("--onnx-model", default="models/audio/export/audio_baseline_v2.onnx")
    parser.add_argument("--config", default="configs/audio_formal_50_v2.yaml")
    parser.add_argument("--manifest", default="data/processed/audio_features_v2/audio_manifest.csv")
    parser.add_argument("--output-dir", default="models/audio/export")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration-limit", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_feature(path: Path) -> np.ndarray:
    with np.load(path) as data:
        feature = data["feature"].astype(np.float32)
    return feature[np.newaxis, np.newaxis, :, :]


def resolve_feature_paths(manifest: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    output = manifest.copy()
    feature_root = manifest_path.parent
    output["feature_path"] = output["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    return output


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate_onnx(model_path: Path, rows: pd.DataFrame, threshold: float) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    y_true: list[int] = []
    y_pred: list[int] = []
    cough_probs: list[float] = []
    for row in rows.itertuples(index=False):
        feature = load_feature(Path(row.feature_path))
        logits = session.run([output_name], {input_name: feature})[0]
        probs = softmax(logits)
        cough_prob = float(probs[0, 1])
        y_true.append(int(row.label_id))
        y_pred.append(1 if cough_prob >= threshold else 0)
        cough_probs.append(cough_prob)
    metrics = binary_metrics(y_true, y_pred)
    metrics["rows"] = len(y_true)
    metrics["threshold"] = threshold
    metrics["mean_cough_prob"] = float(np.mean(cough_probs)) if cough_probs else 0.0
    return metrics


class LogMelCalibrationReader:
    def __init__(self, rows: pd.DataFrame, input_name: str):
        self.rows = rows.reset_index(drop=True)
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.rows):
            return None
        path = Path(str(self.rows.iloc[self.index]["feature_path"]))
        self.index += 1
        return {self.input_name: load_feature(path)}

    def rewind(self) -> None:
        self.index = 0


def onnx_tensor_dtype(model_path: Path) -> tuple[str, str]:
    import onnx

    model = onnx.load(str(model_path))
    dtype_names = {
        1: "float32",
        2: "uint8",
        3: "int8",
        6: "int32",
        7: "int64",
        10: "float16",
    }
    input_dtype = dtype_names.get(model.graph.input[0].type.tensor_type.elem_type, "unknown")
    output_dtype = dtype_names.get(model.graph.output[0].type.tensor_type.elem_type, "unknown")
    return input_dtype, output_dtype


def read_selected_threshold(default: float = 0.75) -> float:
    path = ROOT / "models" / "audio" / "selected" / "selected_threshold.json"
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("threshold", default))


def main() -> int:
    args = parse_args()
    onnx_model = resolve_path(args.onnx_model)
    config_path = resolve_path(args.config)
    manifest_path = resolve_path(args.manifest)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    if not onnx_model.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    config = load_config(config_path)
    manifest = resolve_feature_paths(pd.read_csv(manifest_path), manifest_path)
    calibration_rows = manifest[manifest["split"].isin(["train", "val"])].reset_index(drop=True)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    if calibration_rows.empty:
        raise ValueError("No train/val rows available for representative calibration dataset")
    if test_rows.empty:
        raise ValueError("No test rows available for quantized evaluation")
    if args.calibration_limit:
        calibration_rows = calibration_rows.head(args.calibration_limit)

    threshold = float(args.threshold if args.threshold is not None else read_selected_threshold())
    tflite_path = output_dir / "audio_baseline_v2_int8.tflite"
    int8_onnx_path = output_dir / "audio_baseline_v2_int8.onnx"
    report_path = output_dir / "quantization_report.json"

    tflite_available = has_module("tensorflow") and (
        has_module("onnx_tf") or has_module("onnx2tf")
    )
    conversion_note = None
    quantized_artifact = int8_onnx_path
    quantization_format = "onnx_int8_static_qdq"

    if tflite_available:
        conversion_note = (
            "TensorFlow/TFLite conversion tooling appears installed, but direct ONNX-to-TFLite "
            "conversion is not implemented in this project script yet."
        )
    else:
        conversion_note = (
            "Direct ONNX-to-TFLite int8 export is unavailable in this environment because "
            "TensorFlow plus an ONNX conversion bridge such as onnx-tf or onnx2tf is not installed. "
            "Exported static int8 ONNX as the intermediate deployment candidate instead."
        )

    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    input_name = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    reader = LogMelCalibrationReader(calibration_rows, input_name)
    quantize_static(
        model_input=str(onnx_model),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    float_metrics = evaluate_onnx(onnx_model, test_rows, threshold)
    int8_metrics = evaluate_onnx(int8_onnx_path, test_rows, threshold)
    metric_drop = {
        key: float(float_metrics[key]) - float(int8_metrics[key])
        for key in ("accuracy", "precision", "recall", "f1")
    }
    warning = None
    if metric_drop["f1"] > 0.03:
        warning = f"int8 f1 dropped by {metric_drop['f1']:.6f}, which exceeds 0.03"

    input_dtype, output_dtype = onnx_tensor_dtype(int8_onnx_path)
    report = {
        "ok": warning is None,
        "config": str(config_path),
        "manifest": str(manifest_path),
        "float_onnx_model": str(onnx_model),
        "requested_tflite_output": str(tflite_path),
        "quantized_model": str(quantized_artifact),
        "quantization_format": quantization_format,
        "conversion_note": conversion_note,
        "representative_dataset": {
            "splits": ["train", "val"],
            "rows": int(len(calibration_rows)),
        },
        "test_dataset": {
            "split": "test",
            "rows": int(len(test_rows)),
        },
        "threshold": threshold,
        "float_metrics": float_metrics,
        "int8_metrics": int8_metrics,
        "metric_drop": metric_drop,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "input_scale": None,
        "input_zero_point": None,
        "output_scale": None,
        "output_zero_point": None,
        "warning": warning,
    }
    save_json(report, report_path)

    print(conversion_note)
    print(f"OK: wrote int8 ONNX model to {int8_onnx_path}")
    print(f"OK: wrote quantization report to {report_path}")
    print(f"float_f1={float_metrics['f1']:.6f} int8_f1={int8_metrics['f1']:.6f} drop={metric_drop['f1']:.6f}")
    if warning:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
