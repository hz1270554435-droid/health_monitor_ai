from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.sampling import build_weighted_sampler
from src.common.config import load_config
from src.common.io import ensure_dir


TARGET_ROLES = ["fp_hard_negative", "mid_negative", "fn_positive"]
CLASS_ORDER = ["non_cough", "cough"]
NOT_EXECUTED = [
    "one-epoch training",
    "formal training",
    "model export",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test FP-fix weighted batch sampling without training.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_sampling_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_sampling_summary.md")
    parser.add_argument("--batch-count", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def bool_config(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "y", "on"}


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype(str).value_counts(dropna=False).sort_index().items()
    }


def counter_to_counts(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def proportions(counts: dict[str, int], total: int) -> dict[str, float]:
    denom = max(int(total), 1)
    return {str(key): float(value) / denom for key, value in counts.items()}


def stats(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "max": float(numeric.max()),
    }


def frontend_check(config: dict[str, Any]) -> dict[str, Any]:
    audio = config["audio"]
    labels = config["labels"]
    checks = {
        "frontend": audio.get("frontend") == "board_htk_no_norm_v1",
        "sample_rate": int(audio.get("sample_rate")) == 16000,
        "mono": True,
        "window_seconds": float(audio.get("window_seconds")) == 1.0,
        "hop_seconds": float(audio.get("hop_seconds")) == 0.5,
        "n_mels": int(audio.get("n_mels")) == 40,
        "mel_scale_htk": str(audio.get("mel_scale")).strip().lower() == "htk" and bool(audio.get("htk")) is True,
        "normalize_none": str(audio.get("normalize")).strip().lower() == "none",
        "class_order": list(labels.get("class_order", [])) == CLASS_ORDER,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "mono_policy": "training feature loader uses librosa mono=True",
    }


def validate_weight_column(frame: pd.DataFrame, weight_column: str) -> list[str]:
    errors: list[str] = []
    if weight_column not in frame.columns:
        return [f"Missing weight column: {weight_column}"]
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    if weights.isna().any():
        errors.append(f"Weight column has non-numeric rows: {int(weights.isna().sum())}")
    if (weights <= 0).any():
        errors.append(f"Weight column has <=0 rows: {int((weights <= 0).sum())}")
    return errors


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3.1 FP-fix Sampling Smoke",
        "",
        f"- status: `{report['status']}`",
        f"- sample_weight_active: `{report['sample_weight_active']}`",
        f"- sampler_type: `{report['sampler_type']}`",
        f"- train_split_uses_weighted_sampler: `{report['train_split_uses_weighted_sampler']}`",
        f"- val_split_uses_weighted_sampler: `{report['val_split_uses_weighted_sampler']}`",
        f"- test_split_uses_weighted_sampler: `{report['test_split_uses_weighted_sampler']}`",
        f"- weight_column: `{report['weight_column']}`",
        f"- batch_count_requested: `{report['batch_count_requested']}`",
        f"- batch_count_observed: `{report['batch_count_observed']}`",
        f"- batch_size: `{report['batch_size']}`",
        f"- sampled_rows: `{report['sampled_rows']}`",
        f"- one_epoch_smoke_run: `{report['one_epoch_smoke_run']}`",
        "",
        "## Weight Stats",
        "",
        f"`{report['sample_weight_stats']}`",
        "",
        "## Manifest Train Distribution",
        "",
        f"- labels: `{report['manifest_train_label_distribution']}`",
        f"- mining_role: `{report['manifest_train_mining_role_distribution']}`",
        "",
        "## Sampled Batch Distribution",
        "",
        f"- labels: `{report['sampled_batch_label_distribution']}`",
        f"- mining_role: `{report['sampled_batch_mining_role_distribution']}`",
        "",
        "## Target Role Lift",
        "",
        f"`{report['target_role_lift']}`",
        "",
        "## Errors",
        "",
    ]
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] if errors else ["- none"])
    lines.extend(["", "## Not Executed", ""])
    lines.extend([f"- {item}" for item in report["not_executed"]])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    labels_path = resolve_path(args.labels or config["paths"]["labels_csv"])
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)
    labels = pd.read_csv(labels_path, keep_default_na=False)
    train = labels[labels["split"].astype(str) == "train"].copy().reset_index(drop=True)
    val = labels[labels["split"].astype(str) == "val"].copy()
    test = labels[labels["split"].astype(str) == "test"].copy()

    sampling = config.get("sampling", {})
    training = config.get("train") or config["training"]
    weight_column = str(sampling.get("weight_column", "sample_weight")).strip() or "sample_weight"
    batch_size = int(args.batch_size or training["batch_size"])
    seed = int(args.seed if args.seed is not None else config["project"].get("seed", 42))
    errors: list[str] = []
    errors.extend(validate_weight_column(train, weight_column))

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("torch is required for sampling smoke") from exc

    torch.manual_seed(seed)

    class RowIndexDataset(Dataset):
        def __init__(self, row_count: int):
            self.row_count = int(row_count)

        def __len__(self) -> int:
            return self.row_count

        def __getitem__(self, index: int):
            return int(index)

    sampler = None
    sampler_error = ""
    try:
        sampler = build_weighted_sampler(train, sampling, torch)
    except Exception as exc:
        sampler_error = str(exc)
        errors.append(f"Sampler build failed: {sampler_error}")

    mode = str(sampling.get("mode", "")).strip().lower()
    use_sample_weight = bool_config(sampling.get("use_sample_weight"), default=False)
    sample_weight_active = bool(
        sampler is not None
        and mode == "weighted_random"
        and use_sample_weight
        and not validate_weight_column(train, weight_column)
    )
    train_uses_weighted_sampler = sampler is not None
    val_uses_weighted_sampler = False
    test_uses_weighted_sampler = False

    sampled_label_counter: Counter[str] = Counter()
    sampled_role_counter: Counter[str] = Counter()
    batch_count_observed = 0
    sampled_rows = 0
    if sample_weight_active:
        loader = DataLoader(
            RowIndexDataset(len(train)),
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
        )
        for batch_indices in loader:
            indices = [int(value) for value in batch_indices.tolist()]
            batch_rows = train.iloc[indices]
            sampled_label_counter.update(batch_rows["label"].astype(str).tolist())
            sampled_role_counter.update(batch_rows["mining_role"].astype(str).tolist())
            batch_count_observed += 1
            sampled_rows += len(indices)
            if batch_count_observed >= int(args.batch_count):
                break
    else:
        errors.append("sample_weight_active is false; batch sampling smoke was not executed")

    manifest_label_counts = value_counts(train, "label")
    manifest_role_counts = value_counts(train, "mining_role")
    sampled_label_counts = counter_to_counts(sampled_label_counter)
    sampled_role_counts = counter_to_counts(sampled_role_counter)
    manifest_role_props = proportions(manifest_role_counts, len(train))
    sampled_role_props = proportions(sampled_role_counts, sampled_rows)
    target_role_lift: dict[str, dict[str, Any]] = {}
    for role in TARGET_ROLES:
        manifest_prop = float(manifest_role_props.get(role, 0.0))
        sampled_prop = float(sampled_role_props.get(role, 0.0))
        target_role_lift[role] = {
            "manifest_proportion": manifest_prop,
            "sampled_proportion": sampled_prop,
            "lift": sampled_prop / manifest_prop if manifest_prop > 0 else None,
            "increased": sampled_prop > manifest_prop,
        }
        if manifest_prop > 0 and sampled_prop <= manifest_prop:
            errors.append(f"Sampled proportion did not increase for {role}")

    frontend = frontend_check(config)
    if frontend["status"] != "passed":
        errors.append("board_htk_no_norm_v1 frontend check failed")
    if not train_uses_weighted_sampler:
        errors.append("Train split did not use WeightedRandomSampler")
    if val_uses_weighted_sampler or test_uses_weighted_sampler:
        errors.append("Val/test split unexpectedly used weighted sampler")

    report = {
        "status": "passed" if not errors else "failed",
        "paths": {
            "config": str(config_path),
            "labels": str(labels_path),
            "report": str(report_path),
            "summary_md": str(summary_path),
        },
        "sample_weight_active": bool(sample_weight_active),
        "sampler_type": type(sampler).__name__ if sampler is not None else "",
        "sampler_error": sampler_error,
        "train_split_uses_weighted_sampler": bool(train_uses_weighted_sampler),
        "val_split_uses_weighted_sampler": bool(val_uses_weighted_sampler),
        "test_split_uses_weighted_sampler": bool(test_uses_weighted_sampler),
        "weight_column": weight_column,
        "sampling_config": sampling,
        "sample_weight_stats": stats(train[weight_column]) if weight_column in train.columns else {},
        "rows": {
            "manifest": int(len(labels)),
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "manifest_train_label_distribution": manifest_label_counts,
        "manifest_train_mining_role_distribution": manifest_role_counts,
        "manifest_train_label_proportion": proportions(manifest_label_counts, len(train)),
        "manifest_train_mining_role_proportion": manifest_role_props,
        "sampled_batch_label_distribution": sampled_label_counts,
        "sampled_batch_mining_role_distribution": sampled_role_counts,
        "sampled_batch_label_proportion": proportions(sampled_label_counts, sampled_rows),
        "sampled_batch_mining_role_proportion": sampled_role_props,
        "target_role_lift": target_role_lift,
        "batch_count_requested": int(args.batch_count),
        "batch_count_observed": int(batch_count_observed),
        "batch_size": int(batch_size),
        "sampled_rows": int(sampled_rows),
        "seed": seed,
        "frontend_check": frontend,
        "one_epoch_smoke_run": False,
        "not_executed": NOT_EXECUTED,
        "p01_policy": "board_live_data_1 addon remains train-only/regression-check data, not a generalization set",
        "errors": errors,
    }
    write_json(report, report_path)
    write_summary(report, summary_path)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
