from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.dataset import FeatureDataset
from src.audio.models import build_model
from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.common.metrics import (
    compute_binary_metrics,
    save_confusion_matrix,
    write_classification_report,
)
from src.common.splits import assign_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MIC cough/non_cough model.")
    parser.add_argument("--config", default="configs/audio_baseline.yaml")
    parser.add_argument("--run-dir", required=True, help="Training/evaluation result directory to write outputs into.")
    parser.add_argument("--model-path", required=True, help="Path to best_model.pt.")
    parser.add_argument("--manifest", default=None, help="Defaults to paths.processed_dir/audio_manifest.csv from config.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None, help="Optional cough probability threshold.")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for evaluation") from exc
    return torch, DataLoader


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_manifest(config: dict[str, Any], manifest_arg: str | None) -> tuple[pd.DataFrame, Path]:
    manifest_path = resolve_path(
        manifest_arg or (ROOT / config["paths"]["processed_dir"] / "audio_manifest.csv")
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest has no rows: {manifest_path}")

    split_cfg = config["split"]
    if "split" not in manifest.columns:
        manifest = assign_group_splits(
            manifest,
            group_column=split_cfg["group_column"],
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            seed=int(config["project"]["seed"]),
        )
    return manifest, manifest_path


def resolve_feature_paths(manifest: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    output = manifest.copy()
    feature_root = manifest_path.parent
    output["feature_path"] = output["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    return output


def sample_id(row: pd.Series) -> str:
    if "sample_id" in row and pd.notna(row["sample_id"]):
        return str(row["sample_id"])
    return f"{row['clip_id']}_{int(row['window_index']):04d}"


def threshold_suffix(threshold: float) -> str:
    return f"threshold_{threshold:.2f}".replace(".", "p")


def get_training_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train") or config["training"]


def resolve_model_name(config: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        training = get_training_section(checkpoint_config)
        if training.get("model"):
            return str(training["model"]).strip().lower()
    return str(get_training_section(config).get("model", "small_cnn")).strip().lower()


def main() -> int:
    args = parse_args()
    torch, DataLoader = require_torch()
    config = load_config(args.config)

    run_dir = ensure_dir(resolve_path(args.run_dir))
    model_path = resolve_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")

    saved_manifest = run_dir / "split_manifest.csv"
    manifest_arg = args.manifest or (str(saved_manifest) if saved_manifest.exists() else None)
    manifest, manifest_path = load_manifest(config, manifest_arg)
    manifest = resolve_feature_paths(manifest, manifest_path)
    split_rows = manifest[manifest["split"] == args.split].reset_index(drop=True)
    if split_rows.empty:
        raise ValueError(f"No rows available for split={args.split} in {manifest_path}")

    training = get_training_section(config)
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get(
        "class_names",
        [config["labels"]["negative"], config["labels"]["positive"]],
    )
    resolved_model_name = resolve_model_name(config, checkpoint)
    model = build_model(resolved_model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_model_compatible = True
    model.eval()

    dataset = FeatureDataset(split_rows, root=manifest_path.parent).unwrap()
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
    )

    cough_label = config["labels"]["positive"]
    if cough_label not in class_names:
        raise ValueError(f"Checkpoint class_names does not include cough label '{cough_label}': {class_names}")
    cough_index = int(class_names.index(cough_label))

    y_true: list[int] = []
    y_pred: list[int] = []
    cough_probs: list[float] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            if args.threshold is None:
                predictions = probs.argmax(dim=1).tolist()
            else:
                predictions = [1 if float(row[cough_index]) >= args.threshold else 0 for row in probs]
            y_pred.extend(int(value) for value in predictions)
            y_true.extend(int(value) for value in labels.tolist())
            cough_probs.extend(float(row[cough_index]) for row in probs)

    metrics = {
        "split": args.split,
        "rows": int(len(split_rows)),
        "model_path": str(model_path),
        "manifest": str(manifest_path),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "resolved_model_name": resolved_model_name,
        "checkpoint_model_compatible": checkpoint_model_compatible,
        **compute_binary_metrics(y_true, y_pred),
    }
    if args.threshold is not None:
        metrics["threshold"] = float(args.threshold)
    if bool(config.get("debug", {}).get("small_data", False)):
        metrics["debug_small_data"] = True

    save_json(metrics, run_dir / "metrics.json")
    write_classification_report(y_true, y_pred, class_names, run_dir / "classification_report.txt")
    if args.threshold is not None:
        report_path = run_dir / "classification_report.txt"
        report = report_path.read_text(encoding="utf-8")
        report_path.write_text(f"threshold: {args.threshold:.2f}\n\n{report}", encoding="utf-8")
        suffix = threshold_suffix(float(args.threshold))
        confusion_path = run_dir / f"confusion_matrix_{suffix}.png"
        misclassified_path = run_dir / f"misclassified_{suffix}.csv"
    else:
        confusion_path = run_dir / "confusion_matrix.png"
        misclassified_path = run_dir / "misclassified.csv"
    save_confusion_matrix(y_true, y_pred, class_names, confusion_path)

    eval_rows = split_rows.copy()
    eval_rows["true_id"] = y_true
    eval_rows["pred_id"] = y_pred
    eval_rows["cough_prob"] = cough_probs
    eval_rows["sample_id"] = eval_rows.apply(sample_id, axis=1)
    eval_rows["true_label"] = eval_rows["true_id"].map(lambda value: class_names[int(value)])
    eval_rows["pred_label"] = eval_rows["pred_id"].map(lambda value: class_names[int(value)])
    misclassified = eval_rows[eval_rows["true_id"] != eval_rows["pred_id"]].copy()
    misclassified["start_time"] = misclassified["window_start"]
    misclassified["end_time"] = misclassified["window_end"]
    misclassified[
        [
            "sample_id",
            "feature_path",
            "audio_file",
            "person_id",
            "true_label",
            "pred_label",
            "cough_prob",
            "start_time",
            "end_time",
        ]
    ].to_csv(misclassified_path, index=False)

    print(f"OK: evaluated {len(split_rows)} {args.split} rows; saved results to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
