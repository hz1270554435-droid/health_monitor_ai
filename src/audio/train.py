from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.audio.dataset import FeatureDataset
from src.audio.models import build_model
from src.audio.sampling import build_weighted_sampler, loss_class_weights
from src.common.io import ensure_dir, save_json, timestamp_name
from src.common.metrics import compute_binary_metrics, save_confusion_matrix, write_classification_report
from src.common.splits import assign_group_splits


ROOT = Path(__file__).resolve().parents[2]


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required. Install dependencies with requirements.txt") from exc
    return torch, DataLoader


def set_seed(seed: int, torch_module=None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)


def resolve_path(path: str | Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def task_class_names(config: dict[str, Any]) -> list[str]:
    labels = config["labels"]
    return list(labels.get("class_order") or [labels["negative"], labels["positive"]])


def training_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train") or config["training"]


def primary_metric_name(config: dict[str, Any]) -> str:
    metric = str(config.get("eval", {}).get("primary_metric", "val_f1"))
    return metric if metric.startswith("val_") else f"val_{metric}"


def early_stopping_patience(config: dict[str, Any]) -> int | None:
    value = training_section(config).get("early_stopping_patience")
    if value is None:
        return None
    patience = int(value)
    return patience if patience > 0 else None


def prepare_manifest(manifest_path: str | Path, config: dict[str, Any], root: Path = ROOT) -> pd.DataFrame:
    path = resolve_path(manifest_path, root)
    manifest = pd.read_csv(path)
    if manifest.empty:
        raise ValueError(f"Manifest has no rows: {path}")
    feature_root = path.parent
    manifest["feature_path"] = manifest["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    split_cfg = config["split"]
    group_column = split_cfg.get("group_column", "split_group" if "split_group" in manifest.columns else "person_id")
    if "split" in manifest.columns:
        expected_splits = {"train", "val", "test"}
        split_values = set(manifest["split"].dropna().astype(str))
        invalid_splits = sorted(split_values - expected_splits)
        if invalid_splits:
            raise ValueError(f"Invalid split values in manifest: {invalid_splits}; allowed={sorted(expected_splits)}")
        missing_splits = sorted(expected_splits - split_values)
        if missing_splits:
            raise ValueError(f"Manifest split column is missing required splits: {missing_splits}")
        leaking_groups = manifest.groupby(group_column)["split"].nunique(dropna=True)
        leaking_groups = leaking_groups[leaking_groups > 1]
        if not leaking_groups.empty:
            raise ValueError(
                f"{group_column} appears in multiple splits: {sorted(leaking_groups.index.astype(str))}"
            )
        return manifest

    return assign_group_splits(
        manifest,
        group_column=group_column,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(config["project"]["seed"]),
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
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


def write_config_copy(config: dict[str, Any], config_path: str | Path | None, output_dir: Path) -> None:
    if config_path is not None:
        import shutil

        shutil.copy2(resolve_path(config_path), output_dir / "config.yaml")
        return
    import yaml

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_training(
    config: dict[str, Any],
    config_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    root: Path = ROOT,
) -> dict[str, Path]:
    torch, DataLoader = require_torch()
    set_seed(int(config["project"]["seed"]), torch_module=torch)

    manifest_file = resolve_path(
        manifest_path or (root / config["paths"]["processed_dir"] / "audio_manifest.csv"),
        root,
    )
    manifest = prepare_manifest(manifest_file, config, root=root)
    if manifest["label_id"].nunique() < 2:
        raise ValueError("Training requires at least two label_id classes in the manifest")

    name = run_name or timestamp_name(config["run"]["name"])
    results_dir = ensure_dir(resolve_path(config["paths"]["results_dir"], root) / name)
    models_dir = ensure_dir(resolve_path(config["paths"]["models_dir"], root) / name)
    write_config_copy(config, config_path, results_dir)
    split_manifest_path = results_dir / "split_manifest.csv"
    manifest.to_csv(split_manifest_path, index=False)

    training = training_section(config)
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    loaders = {}
    for split in ("train", "val", "test"):
        split_rows = manifest[manifest["split"] == split]
        if split_rows.empty:
            raise ValueError(f"Split has no rows: {split}")
        dataset = FeatureDataset(split_rows, root=manifest_file.parent).unwrap()
        sampler = build_weighted_sampler(split_rows, config.get("sampling"), torch) if split == "train" else None
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=int(training.get("num_workers", 0)),
        )

    class_names = task_class_names(config)
    model_name = str(training.get("model", "small_cnn"))
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    weight = loss_class_weights(manifest[manifest["split"] == "train"], training, torch, device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer_name = str(training.get("optimizer", "adamw")).strip().lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name in {"adamw", "adam"} else torch.optim.AdamW
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    total_epochs = int(epochs or training["epochs"])
    monitor_metric = primary_metric_name(config)
    patience = early_stopping_patience(config)
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = models_dir / "best_model.pt"
    train_log_rows: list[dict[str, float | int | bool | str]] = []

    for epoch in range(1, total_epochs + 1):
        train_loss = run_epoch(model, loaders["train"], criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, loaders["val"], criterion, optimizer, device, train=False)
        y_true, y_pred = predict(model, loaders["val"], device)
        val_metrics = compute_binary_metrics(y_true, y_pred)
        val_metric_values = {f"val_{key}": value for key, value in val_metrics.items()}
        if monitor_metric not in val_metric_values:
            raise ValueError(f"Unsupported primary metric: {monitor_metric}; available={sorted(val_metric_values)}")
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
                    "model_name": model_name,
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
        "split_counts": manifest["split"].value_counts().to_dict(),
        "test": test_metrics,
    }
    save_json(metrics, results_dir / "metrics.json")
    save_json({str(index): label for index, label in enumerate(class_names)}, results_dir / "label_map.json")
    write_classification_report(y_true, y_pred, class_names, results_dir / "classification_report.txt")
    save_confusion_matrix(y_true, y_pred, class_names, results_dir / "confusion_matrix.png")
    print(f"OK: saved results to {results_dir}")
    return {
        "results_dir": results_dir,
        "models_dir": models_dir,
        "checkpoint_path": checkpoint_path,
        "split_manifest_path": split_manifest_path,
    }
