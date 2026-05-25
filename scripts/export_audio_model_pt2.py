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


DUMMY_SHAPE = (1, 1, 40, 101)
MAX_ABS_DIFF_THRESHOLD = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected MIC audio PyTorch checkpoint to PT2.")
    parser.add_argument("--config", default="configs/audio_formal_50_v2.yaml")
    parser.add_argument(
        "--checkpoint",
        "--model-path",
        dest="checkpoint",
        default="models/audio/selected/audio_baseline_v2_best_model.pt",
        help="Path to selected PyTorch checkpoint.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        default="models/audio/export/audio_baseline_v2.pt2",
        help="Output PT2 path.",
    )
    parser.add_argument(
        "--report",
        default="models/audio/export/export_report_pt2.json",
        help="Output export report JSON path.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Optional deployment metadata JSON path. Defaults to <output-dir>/metadata_pt2.json.",
    )
    parser.add_argument(
        "--onnx-model",
        default=None,
        help="Optional ONNX model path to verify the PT2 export against the board deployment ONNX.",
    )
    parser.add_argument(
        "--threshold-sweep",
        default=None,
        help="Optional threshold_sweep.csv path. Defaults to the run results directory when available.",
    )
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for PT2 export") from exc
    if not hasattr(torch, "export"):
        raise RuntimeError("This torch installation does not provide torch.export")
    return torch


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required when --onnx-model is provided. "
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


def tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def first_tensor(output: Any) -> Any:
    if isinstance(output, tuple):
        if len(output) != 1:
            raise ValueError(f"Expected single tensor output, got tuple length={len(output)}")
        return output[0]
    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(f"Expected single tensor output, got list length={len(output)}")
        return output[0]
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        if len(output) == 1:
            return next(iter(output.values()))
        raise ValueError(f"Expected single tensor output, got dict keys={sorted(output)}")
    return output


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


def infer_feature_shape(config: dict[str, Any]) -> tuple[int, int, Path | None]:
    manifest_path = ROOT / config["paths"]["processed_dir"] / "audio_manifest.csv"
    if not manifest_path.exists():
        return DUMMY_SHAPE[2], DUMMY_SHAPE[3], None

    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        return DUMMY_SHAPE[2], DUMMY_SHAPE[3], None
    feature_path = Path(str(manifest.iloc[0]["feature_path"]))
    if not feature_path.is_absolute():
        feature_path = manifest_path.parent / feature_path
    if not feature_path.exists():
        return DUMMY_SHAPE[2], DUMMY_SHAPE[3], None

    with np.load(feature_path) as data:
        feature = data["feature"]
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={feature.shape} from {feature_path}")
    return int(feature.shape[0]), int(feature.shape[1]), feature_path


def main() -> int:
    args = parse_args()
    torch = require_torch()

    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = resolve_path(args.output)
    report_path = resolve_path(args.report)
    metadata_path = resolve_path(args.metadata) if args.metadata else output_path.parent / "metadata_pt2.json"
    onnx_path = resolve_path(args.onnx_model) if args.onnx_model else None

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if onnx_path is not None and not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    config = load_config(config_path)
    n_mels, time_frames, feature_path = infer_feature_shape(config)
    expected_n_mels = int(config["audio"]["n_mels"])
    if n_mels != expected_n_mels:
        raise ValueError(
            f"Feature n_mels={n_mels} does not match config audio.n_mels={expected_n_mels}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint.get("class_names", [config["labels"]["negative"], config["labels"]["positive"]])
    if len(class_names) < 2:
        raise ValueError(f"Expected at least 2 class names, got {len(class_names)}")

    model_name = str(checkpoint.get("model_name", config.get("training", {}).get("model", "small_cnn")))
    model = build_model(model_name, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dummy_shape = (1, 1, n_mels, time_frames)
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    dummy_input = torch.randn(dummy_shape, generator=generator, dtype=torch.float32)

    ensure_dir(output_path.parent)
    with torch.no_grad():
        pytorch_output = first_tensor(model(dummy_input)).detach().cpu()

    exported_program = torch.export.export(model, (dummy_input,))
    torch.export.save(exported_program, output_path)

    loaded_program = torch.export.load(output_path)
    loaded_module = loaded_program.module()
    with torch.no_grad():
        pt2_output = first_tensor(loaded_module(dummy_input)).detach().cpu()

    abs_diff = (pytorch_output - pt2_output).abs()
    max_abs_diff = float(abs_diff.max().item())
    mean_abs_diff = float(abs_diff.mean().item())
    ok = bool(max_abs_diff < MAX_ABS_DIFF_THRESHOLD)

    onnx_output = None
    onnx_max_abs_diff = None
    onnx_mean_abs_diff = None
    onnx_allclose = None
    if onnx_path is not None:
        ort = require_onnxruntime()
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_output_array = session.run(None, {"input": dummy_input.detach().cpu().numpy()})[0]
        onnx_output = torch.as_tensor(onnx_output_array)
        onnx_abs_diff = (pt2_output - onnx_output).abs()
        onnx_max_abs_diff = float(onnx_abs_diff.max().item())
        onnx_mean_abs_diff = float(onnx_abs_diff.mean().item())
        onnx_allclose = bool(np.allclose(pt2_output.numpy(), onnx_output_array, rtol=1e-4, atol=1e-5))
        ok = bool(ok and onnx_allclose)

    selected_threshold_path = checkpoint_path.parent / "selected_threshold.json"
    label_map_path = checkpoint_path.parent / "label_map.json"
    threshold_sweep_path = (
        resolve_path(args.threshold_sweep)
        if args.threshold_sweep
        else default_threshold_sweep_path(config)
    )
    threshold_summary = read_threshold_summary(threshold_sweep_path)
    frontend = str(config.get("audio", {}).get("frontend", ""))
    run_name = str(config.get("run", {}).get("name", ""))
    class_order = list(config.get("labels", {}).get("class_order", class_names))
    deploy_model_name = deployment_model_name(run_name, model_name)
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
        "checkpoint_path": str(checkpoint_path),
        "pt2_path": str(output_path),
        "onnx_path": str(onnx_path) if onnx_path is not None else None,
    }
    report = {
        "ok": ok,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "run_name": run_name,
        "frontend": frontend,
        "pt2_path": str(output_path),
        "onnx_path": str(onnx_path) if onnx_path is not None else None,
        "input_shape": list(dummy_shape),
        "output_shape": list(pytorch_output.shape),
        "feature_shape_source": str(feature_path) if feature_path is not None else None,
        "class_names": list(class_names),
        "class_order": class_order,
        "recommended_thresholds": threshold_summary,
        "selected_threshold": read_json_if_exists(selected_threshold_path),
        "label_map": read_json_if_exists(label_map_path),
        "pytorch_logits": tensor_to_list(pytorch_output),
        "pt2_logits": tensor_to_list(pt2_output),
        "onnx_logits": tensor_to_list(onnx_output) if onnx_output is not None else None,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "pt2_vs_onnx_max_abs_diff": onnx_max_abs_diff,
        "pt2_vs_onnx_mean_abs_diff": onnx_mean_abs_diff,
        "pt2_vs_onnx_allclose": onnx_allclose,
        "max_abs_diff_threshold": MAX_ABS_DIFF_THRESHOLD,
    }
    save_json(report, report_path)
    save_json(metadata, metadata_path)

    if not ok:
        raise RuntimeError(
            f"PT2 output mismatch: max_abs_diff={max_abs_diff:.8f}, "
            f"mean_abs_diff={mean_abs_diff:.8f}, "
            f"pt2_vs_onnx_max_abs_diff={onnx_max_abs_diff}; report={report_path}"
        )

    print(f"OK: exported PT2 to {output_path}")
    print(f"OK: verified PT2 output; max_abs_diff={max_abs_diff:.8f}, mean_abs_diff={mean_abs_diff:.8f}")
    if onnx_path is not None:
        print(
            "OK: verified PT2 vs ONNXRuntime output; "
            f"max_abs_diff={onnx_max_abs_diff:.8f}, mean_abs_diff={onnx_mean_abs_diff:.8f}"
        )
    print(f"OK: wrote report to {report_path}")
    print(f"OK: wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
