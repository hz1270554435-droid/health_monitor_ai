from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.audio.dataset import FeatureDataset
from src.audio.models import build_model
from src.audio.train import ROOT, prepare_manifest, resolve_path, task_class_names, training_section
from src.common.io import ensure_dir, save_json
from src.common.metrics import compute_binary_metrics, save_confusion_matrix, write_classification_report


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for evaluation") from exc
    return torch, DataLoader


def probability_column(config: dict[str, Any]) -> str:
    return f"{config['labels']['positive']}_prob"


def sample_id(row: pd.Series) -> str:
    if "sample_id" in row and pd.notna(row["sample_id"]):
        return str(row["sample_id"])
    return f"{row['clip_id']}_{int(row['window_index']):04d}"


def threshold_predictions(probs: list[float], threshold: float) -> list[int]:
    return [1 if prob >= threshold else 0 for prob in probs]


def merge_false_positive_events(false_positive_rows: pd.DataFrame, gap_seconds: float) -> int:
    if false_positive_rows.empty:
        return 0
    required = {"audio_file", "window_start", "window_end"}
    if not required.issubset(false_positive_rows.columns):
        return int(len(false_positive_rows))
    events = 0
    for _, group in false_positive_rows.sort_values(["audio_file", "window_start"]).groupby("audio_file"):
        current_end: float | None = None
        for row in group.itertuples(index=False):
            start = float(getattr(row, "window_start"))
            end = float(getattr(row, "window_end"))
            if current_end is None or start > current_end + gap_seconds:
                events += 1
                current_end = end
            else:
                current_end = max(current_end, end)
    return events


def false_alarm_metrics(eval_rows: pd.DataFrame, gap_seconds: float, night_hours: float) -> dict[str, float | int]:
    negative_rows = eval_rows[eval_rows["true_id"] == 0].copy()
    false_positive_rows = eval_rows[(eval_rows["true_id"] == 0) & (eval_rows["pred_id"] == 1)].copy()
    if {"window_start", "window_end"}.issubset(negative_rows.columns):
        negative_seconds = (
            pd.to_numeric(negative_rows["window_end"], errors="coerce")
            - pd.to_numeric(negative_rows["window_start"], errors="coerce")
        ).clip(lower=0).sum()
    else:
        negative_seconds = 0.0
    negative_hours = float(negative_seconds) / 3600.0
    fp_events = merge_false_positive_events(false_positive_rows, gap_seconds)
    fp_per_hour = float(fp_events / negative_hours) if negative_hours > 0 else 0.0
    return {
        "false_positive_windows": int(len(false_positive_rows)),
        "false_positive_events": int(fp_events),
        "negative_duration_hours": negative_hours,
        "false_positives_per_hour": fp_per_hour,
        "false_alarms_per_night": fp_per_hour * float(night_hours),
        "event_merge_gap_seconds": float(gap_seconds),
        "night_hours": float(night_hours),
    }


def load_model(config: dict[str, Any], model_path: Path, device):
    torch, _ = require_torch()
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", task_class_names(config))
    model_name = checkpoint.get("model_name", training_section(config).get("model", "small_cnn"))
    model = build_model(str(model_name), num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return torch, model, class_names


def collect_probabilities(config: dict[str, Any], model_path: Path, manifest: pd.DataFrame, manifest_path: Path, split: str):
    torch, DataLoader = require_torch()
    training = training_section(config)
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    torch, model, class_names = load_model(config, model_path, device)
    positive_label = config["labels"]["positive"]
    if positive_label not in class_names:
        raise ValueError(f"Checkpoint class_names does not include positive label '{positive_label}': {class_names}")
    positive_index = int(class_names.index(positive_label))

    split_rows = manifest[manifest["split"] == split].reset_index(drop=True)
    if split_rows.empty:
        raise ValueError(f"No rows available for split={split} in {manifest_path}")
    dataset = FeatureDataset(split_rows, root=manifest_path.parent).unwrap()
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
    )

    y_true: list[int] = []
    y_argmax: list[int] = []
    positive_probs: list[float] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            y_argmax.extend(int(value) for value in probs.argmax(dim=1).tolist())
            y_true.extend(int(value) for value in labels.tolist())
            positive_probs.extend(float(row[positive_index]) for row in probs)
    return split_rows, class_names, y_true, y_argmax, positive_probs


def evaluate_checkpoint(
    config: dict[str, Any],
    run_dir: str | Path,
    model_path: str | Path,
    manifest_path: str | Path | None = None,
    split: str = "test",
    threshold: float | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    run_path = ensure_dir(resolve_path(run_dir, root))
    checkpoint_path = resolve_path(model_path, root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    saved_manifest = run_path / "split_manifest.csv"
    manifest_file = resolve_path(manifest_path or (saved_manifest if saved_manifest.exists() else root / config["paths"]["processed_dir"] / "audio_manifest.csv"), root)
    manifest = prepare_manifest(manifest_file, config, root=root)
    split_rows, class_names, y_true, y_argmax, positive_probs = collect_probabilities(
        config, checkpoint_path, manifest, manifest_file, split
    )
    y_pred = threshold_predictions(positive_probs, threshold) if threshold is not None else y_argmax

    prob_col = probability_column(config)
    eval_rows = split_rows.copy()
    eval_rows["true_id"] = y_true
    eval_rows["pred_id"] = y_pred
    eval_rows[prob_col] = positive_probs
    if prob_col != "cough_prob" and config["labels"]["positive"] == "cough":
        eval_rows["cough_prob"] = positive_probs
    eval_rows["sample_id"] = eval_rows.apply(sample_id, axis=1)
    eval_rows["true_label"] = eval_rows["true_id"].map(lambda value: class_names[int(value)])
    eval_rows["pred_label"] = eval_rows["pred_id"].map(lambda value: class_names[int(value)])

    false_alarm_cfg = config.get("eval", {}).get("false_alarm", {})
    fa_metrics = false_alarm_metrics(
        eval_rows,
        gap_seconds=float(false_alarm_cfg.get("event_merge_gap_seconds", 5.0)),
        night_hours=float(false_alarm_cfg.get("night_hours", 8.0)),
    )
    metrics = {
        "split": split,
        "rows": int(len(split_rows)),
        "model_path": str(checkpoint_path),
        "manifest": str(manifest_file),
        "positive_probability_column": prob_col,
        "split_counts": manifest["split"].value_counts().to_dict(),
        **compute_binary_metrics(y_true, y_pred),
        **fa_metrics,
    }
    if threshold is not None:
        metrics["threshold"] = float(threshold)

    suffix = "" if threshold is None else f"_threshold_{threshold:.2f}".replace(".", "p")
    save_json(metrics, run_path / ("metrics.json" if threshold is None else f"metrics{suffix}.json"))
    write_classification_report(y_true, y_pred, class_names, run_path / ("classification_report.txt" if threshold is None else f"classification_report{suffix}.txt"))
    save_confusion_matrix(y_true, y_pred, class_names, run_path / ("confusion_matrix.png" if threshold is None else f"confusion_matrix{suffix}.png"))

    misclassified = eval_rows[eval_rows["true_id"] != eval_rows["pred_id"]].copy()
    misclassified["start_time"] = misclassified["window_start"]
    misclassified["end_time"] = misclassified["window_end"]
    columns = [
        "sample_id",
        "feature_path",
        "audio_file",
        "person_id",
        "true_label",
        "pred_label",
        prob_col,
        "start_time",
        "end_time",
    ]
    misclassified[[column for column in columns if column in misclassified.columns]].to_csv(
        run_path / ("misclassified.csv" if threshold is None else f"misclassified{suffix}.csv"),
        index=False,
    )
    return metrics


def threshold_metrics(y_true: list[int], positive_probs: list[float], threshold: float) -> dict[str, float | int]:
    y_pred = threshold_predictions(positive_probs, threshold)
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    total = len(y_true)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def run_threshold_sweep(
    config: dict[str, Any],
    run_dir: str | Path,
    model_path: str | Path,
    split: str = "test",
    root: Path = ROOT,
) -> pd.DataFrame:
    run_path = ensure_dir(resolve_path(run_dir, root))
    manifest_file = run_path / "split_manifest.csv"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Expected saved split manifest: {manifest_file}")
    manifest = prepare_manifest(manifest_file, config, root=root)
    checkpoint_path = resolve_path(model_path, root)
    _, _, y_true, _, positive_probs = collect_probabilities(config, checkpoint_path, manifest, manifest_file, split)
    rows = [threshold_metrics(y_true, positive_probs, threshold / 100.0) for threshold in range(10, 100, 5)]
    sweep = pd.DataFrame(rows)
    min_recall = float(config.get("eval", {}).get("threshold_min_recall", 0.90))
    candidates = sweep[sweep["recall"] >= min_recall].copy()
    recommended_threshold: float | None = None
    if not candidates.empty:
        candidates = candidates.sort_values(["f1", "threshold"], ascending=[False, True])
        recommended_threshold = float(candidates.iloc[0]["threshold"])
        sweep["recommended"] = sweep["threshold"].eq(recommended_threshold)
    else:
        sweep["recommended"] = False
    sweep.to_csv(run_path / "threshold_sweep.csv", index=False)
    return sweep
