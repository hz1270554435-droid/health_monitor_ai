from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.dataset import FeatureDataset
from src.audio.models import build_model
from src.audio.sampling import build_weighted_sampler
from src.common.config import load_config
from src.common.io import copy_config, ensure_dir, save_json, timestamp_name
from src.common.metrics import (
    compute_binary_metrics,
    save_confusion_matrix,
    write_classification_report,
)
from src.common.splits import assign_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIC cough/non_cough small CNN baseline.")
    parser.add_argument("--config", default="configs/audio_baseline.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required. Install dependencies with requirements.txt") from exc
    return torch, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_manifest(manifest_path: Path, config: dict) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    feature_root = manifest_path.parent
    manifest["feature_path"] = manifest["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    split_cfg = config["split"]
    if "split" in manifest.columns:
        expected_splits = {"train", "val", "test"}
        split_values = set(manifest["split"].dropna().astype(str))
        invalid_splits = sorted(split_values - expected_splits)
        if invalid_splits:
            raise ValueError(f"Invalid split values in manifest: {invalid_splits}; allowed={sorted(expected_splits)}")
        missing_splits = sorted(expected_splits - split_values)
        if missing_splits:
            raise ValueError(f"Manifest split column is missing required splits: {missing_splits}")
        leaking_groups = manifest.groupby(split_cfg["group_column"])["split"].nunique(dropna=True)
        leaking_groups = leaking_groups[leaking_groups > 1]
        if not leaking_groups.empty:
            raise ValueError(
                f"{split_cfg['group_column']} appears in multiple splits: {sorted(leaking_groups.index.astype(str))}"
            )
        return manifest

    return assign_group_splits(
        manifest,
        group_column=split_cfg["group_column"],
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(config["project"]["seed"]),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    import torch

    model.train(train)
    total_loss = 0.0
    total_items = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * int(labels.size(0))
        total_items += int(labels.size(0))
    return total_loss / max(total_items, 1)


def predict(model, loader, device) -> tuple[list[int], list[int]]:
    import torch

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(int(value) for value in predictions)
            y_true.extend(int(value) for value in labels.tolist())
    return y_true, y_pred


def get_training_section(config: dict) -> dict:
    return config.get("train") or config["training"]


def primary_metric_name(config: dict) -> str:
    metric = str(config.get("eval", {}).get("primary_metric", "val_f1"))
    return metric if metric.startswith("val_") else f"val_{metric}"


def early_stopping_patience(config: dict) -> int | None:
    train_cfg = config.get("train", {})
    training_cfg = config.get("training", {})
    value = train_cfg.get("early_stopping_patience", training_cfg.get("early_stopping_patience"))
    if value is None:
        return None
    patience = int(value)
    return patience if patience > 0 else None


def main() -> int:
    args = parse_args()
    torch, DataLoader = require_torch()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))

    manifest_path = Path(args.manifest or (ROOT / config["paths"]["processed_dir"] / "audio_manifest.csv"))
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = prepare_manifest(manifest_path, config)

    run_name = args.run_name or timestamp_name(config["run"]["name"])
    results_dir = ensure_dir(ROOT / config["paths"]["results_dir"] / run_name)
    models_dir = ensure_dir(ROOT / config["paths"]["models_dir"] / run_name)
    copy_config(args.config, results_dir, output_name="config.yaml")
    split_manifest_path = results_dir / "split_manifest.csv"
    manifest.to_csv(split_manifest_path, index=False)

    training = get_training_section(config)
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    loaders = {}
    sampler_status = {}
    for split in ("train", "val", "test"):
        split_rows = manifest[manifest["split"] == split]
        if split_rows.empty:
            raise ValueError(f"Split has no rows: {split}")
        dataset = FeatureDataset(split_rows, root=manifest_path.parent).unwrap()
        sampler = build_weighted_sampler(split_rows, config.get("sampling"), torch) if split == "train" else None
        sampler_status[split] = {
            "uses_weighted_sampler": sampler is not None,
            "sampler_type": type(sampler).__name__ if sampler is not None else None,
            "rows": int(len(split_rows)),
        }
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=int(training.get("num_workers", 0)),
        )

    class_names = list(config["labels"].get("class_order") or [config["labels"]["negative"], config["labels"]["positive"]])
    model_name = str(training.get("model", "small_cnn")).strip().lower()
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    epochs = int(args.epochs or training["epochs"])
    monitor_metric = primary_metric_name(config)
    patience = early_stopping_patience(config)
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = models_dir / "best_model.pt"
    train_log_rows: list[dict[str, float | int | bool]] = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        y_true, y_pred = predict(model, loaders["val"], device)
        val_metrics = compute_binary_metrics(y_true, y_pred)
        val_metric_values = {f"val_{key}": value for key, value in val_metrics.items()}
        if monitor_metric not in val_metric_values:
            raise ValueError(
                f"Unsupported primary metric: {monitor_metric}; "
                f"available={sorted(val_metric_values)}"
            )
        current_metric = float(val_metric_values[monitor_metric])
        is_best = current_metric > best_metric
        early_stop_triggered = False
        if is_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            early_stop_triggered = bool(patience is not None and epochs_without_improvement >= patience)
        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "is_best": is_best,
                "monitor_metric": monitor_metric,
                "monitor_value": current_metric,
                "epochs_without_improvement": epochs_without_improvement,
                "early_stop_triggered": early_stop_triggered,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} {monitor_metric}={current_metric:.4f}"
        )
        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "config": config,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        if early_stop_triggered:
            print(
                f"early stopping triggered at epoch={epoch}; "
                f"best_epoch={best_epoch} {monitor_metric}={best_metric:.4f}"
            )
            break

    pd.DataFrame(train_log_rows).to_csv(results_dir / "train_log.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    y_true, y_pred = predict(model, loaders["test"], device)
    test_metrics = compute_binary_metrics(y_true, y_pred)
    metrics = {
        "best_epoch": best_epoch,
        "primary_metric": monitor_metric,
        "best_metric": best_metric,
        "epochs_run": len(train_log_rows),
        "early_stopping_patience": patience,
        "early_stop_triggered": bool(train_log_rows and train_log_rows[-1]["early_stop_triggered"]),
        "checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "class_names": class_names,
        "sampler": sampler_status,
        "sample_weight_active": bool(
            sampler_status.get("train", {}).get("uses_weighted_sampler")
            and str(config.get("sampling", {}).get("mode", "")).strip().lower() == "weighted_random"
            and bool(config.get("sampling", {}).get("use_sample_weight", False))
        ),
        "weight_column": str(config.get("sampling", {}).get("weight_column", "sample_weight")),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "test": test_metrics,
    }
    save_json(metrics, results_dir / "metrics.json")
    write_classification_report(y_true, y_pred, class_names, results_dir / "classification_report.txt")
    save_confusion_matrix(y_true, y_pred, class_names, results_dir / "confusion_matrix.png")
    print(f"OK: saved results to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
