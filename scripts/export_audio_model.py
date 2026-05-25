from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.models import build_model
from src.common.config import load_config
from src.common.io import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected MIC audio model to ONNX.")
    parser.add_argument("--config", default="configs/audio_formal_50_v2.yaml")
    parser.add_argument(
        "--model-path",
        default="models/audio/selected/audio_baseline_v2_best_model.pt",
        help="Path to selected PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="models/audio/export/audio_baseline_v2.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--report",
        default="models/audio/export/export_report.json",
        help="Output export report JSON path.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional deployment metadata JSON path. Defaults to <output-dir>/metadata.json.",
    )
    parser.add_argument(
        "--threshold-sweep",
        default=None,
        help="Optional threshold_sweep.csv path. Defaults to the run results directory when available.",
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for ONNX export") from exc
    return torch


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required to verify the exported model. "
            "Install it with: python -m pip install onnx onnxruntime"
        ) from exc
    return ort


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def read_threshold_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    import pandas as pd

    sweep = pd.read_csv(path)
    summary: dict[str, Any] = {
        "source": str(path),
        "test_thresholds": [0.50, 0.75, 0.90, 0.95],
    }
    if "recommended" in sweep.columns:
        recommended = sweep[sweep["recommended"].astype(bool)]
        if not recommended.empty:
            row = recommended.iloc[0]
            summary["recommended"] = {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
    return summary


def default_threshold_sweep_path(config: dict[str, Any]) -> Path:
    run_name = str(config.get("run", {}).get("name", ""))
    results_dir = str(config.get("paths", {}).get("results_dir", "results/audio"))
    return ROOT / results_dir / run_name / "threshold_sweep.csv"


def deployment_model_name(run_name: str, checkpoint_model_name: str) -> str:
    if run_name == "audio_baseline_v3_board_htk_hardneg":
        return "audio_model_v3_board_htk_hardneg"
    if run_name == "audio_baseline_v3_1_fpfix":
        return "audio_model_v3_1_fpfix"
    if run_name.startswith("audio_baseline_"):
        return run_name.replace("audio_baseline", "audio_model", 1)
    return checkpoint_model_name


def infer_feature_shape(config: dict[str, Any]) -> tuple[int, int, Path]:
    manifest_path = ROOT / config["paths"]["processed_dir"] / "audio_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest not found: {manifest_path}")

    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Processed manifest is empty: {manifest_path}")
    feature_path = Path(str(manifest.iloc[0]["feature_path"]))
    if not feature_path.is_absolute():
        feature_path = manifest_path.parent / feature_path
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    with np.load(feature_path) as data:
        feature = data["feature"]
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={feature.shape} from {feature_path}")
    return int(feature.shape[0]), int(feature.shape[1]), feature_path


def main() -> int:
    args = parse_args()
    torch = require_torch()
    ort = require_onnxruntime()

    config_path = resolve_path(args.config)
    model_path = resolve_path(args.model_path)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)
    metadata_path = resolve_path(args.metadata) if args.metadata else output_path.parent / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    config = load_config(config_path)
    n_mels, time_frames, feature_path = infer_feature_shape(config)
    expected_n_mels = int(config["audio"]["n_mels"])
    if n_mels != expected_n_mels:
        raise ValueError(f"Feature n_mels={n_mels} does not match config audio.n_mels={expected_n_mels}")

    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint.get("class_names", [config["labels"]["negative"], config["labels"]["positive"]])
    model_name = str(checkpoint.get("model_name", config.get("training", {}).get("model", "small_cnn")))
    model = build_model(model_name, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_shape = (1, 1, n_mels, time_frames)
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    dummy_input = torch.randn(dummy_shape, generator=generator, dtype=torch.float32)

    ensure_dir(output_path.parent)
    with torch.no_grad():
        pytorch_output = model(dummy_input).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.detach().cpu().numpy()})[0]
    abs_diff = np.abs(pytorch_output - onnx_output)
    max_abs_diff = float(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    allclose = bool(np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5))

    selected_threshold_path = model_path.parent / "selected_threshold.json"
    label_map_path = model_path.parent / "label_map.json"
    threshold_sweep_path = (
        resolve_path(args.threshold_sweep)
        if args.threshold_sweep
        else default_threshold_sweep_path(config)
    )
    threshold_summary = read_threshold_summary(threshold_sweep_path)
    frontend = str(config.get("audio", {}).get("frontend", ""))
    run_name = str(config.get("run", {}).get("name", ""))
    deploy_model_name = deployment_model_name(run_name, model_name)
    class_order = list(config.get("labels", {}).get("class_order", class_names))
    metadata = {
        "model_name": deploy_model_name,
        "checkpoint_model_name": model_name,
        "run_name": run_name,
        "frontend": frontend,
        "input_shape": list(dummy_shape),
        "output_shape": list(pytorch_output.shape),
        "class_order": class_order,
        "recommended_thresholds": threshold_summary or {"test_thresholds": [0.50, 0.75, 0.90, 0.95]},
        "training_config_path": str(config_path),
        "checkpoint_path": str(model_path),
        "onnx_path": str(output_path),
    }
    report = {
        "ok": allclose,
        "config": str(config_path),
        "checkpoint": str(model_path),
        "model_name": model_name,
        "run_name": run_name,
        "frontend": frontend,
        "onnx_path": str(output_path),
        "opset": int(args.opset),
        "input_shape": list(dummy_shape),
        "output_shape": list(pytorch_output.shape),
        "feature_shape_source": str(feature_path),
        "class_names": list(class_names),
        "class_order": class_order,
        "recommended_thresholds": threshold_summary,
        "selected_threshold": read_json_if_exists(selected_threshold_path),
        "label_map": read_json_if_exists(label_map_path),
        "pytorch_logits": pytorch_output.tolist(),
        "onnx_logits": onnx_output.tolist(),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "allclose_rtol": 1e-4,
        "allclose_atol": 1e-5,
    }
    save_json(report, report_path)
    save_json(metadata, metadata_path)

    if not allclose:
        raise RuntimeError(
            f"ONNXRuntime output mismatch: max_abs_diff={max_abs_diff:.8f}, "
            f"mean_abs_diff={mean_abs_diff:.8f}; report={report_path}"
        )

    print(f"OK: exported ONNX to {output_path}")
    print(f"OK: verified ONNXRuntime output; max_abs_diff={max_abs_diff:.8f}")
    print(f"OK: wrote report to {report_path}")
    print(f"OK: wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
