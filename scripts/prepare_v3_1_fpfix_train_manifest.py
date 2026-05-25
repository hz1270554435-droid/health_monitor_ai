from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import iter_windows, load_audio_segment, log_mel_feature
from src.common.io import ensure_dir


BOARD_HTK_NO_NORM_V1 = {
    "frontend": "board_htk_no_norm_v1",
    "sample_rate": 16000,
    "window_seconds": 1.0,
    "hop_seconds": 0.5,
    "n_mels": 40,
    "n_fft": 1024,
    "hop_length": 160,
    "fmin": 50,
    "fmax": 7600,
    "mel_scale": "htk",
    "htk": True,
    "center": True,
    "top_db": 80,
    "normalize": "none",
}
EXPECTED_FEATURE_SHAPE = (40, 101)
CLASS_ORDER = ["non_cough", "cough"]

ADDON_REQUIRED_COLUMNS = [
    "label",
    "person_id",
    "session_id",
    "original_path",
    "source_audio",
    "window_start_sec",
    "window_end_sec",
    "manual_decision",
    "manual_subtype_norm",
    "review_bucket",
    "v3_cough_prob",
    "yamnet_top1",
]
ADDON_TRACE_COLUMNS = [
    "source_domain",
    "person_id",
    "session_id",
    "original_path",
    "source_audio",
    "window_start_sec",
    "window_end_sec",
    "manual_decision",
    "manual_subtype_raw",
    "manual_subtype_norm",
    "review_bucket",
    "v3_cough_prob",
    "yamnet_top1",
    "yamnet_cough_score",
    "yamnet_speech_score",
    "yamnet_throat_like_score",
]
BASE_REQUIRED_COLUMNS = [
    "clip_id",
    "session_id",
    "person_id",
    "start_time",
    "end_time",
    "label",
    "audio_file",
    "split",
]
OUTPUT_PRIORITY_COLUMNS = [
    "clip_id",
    "session_id",
    "person_id",
    "start_time",
    "end_time",
    "label",
    "audio_file",
    "split",
    "hard_negative",
    "sample_weight",
    "source",
    "source_domain",
    "change_reason",
    "hard_negative_source",
    "negative_type",
    "mining_role",
    "row_origin",
    "original_path",
    "source_audio",
    "canonical_audio_path",
    "source_start_time",
    "source_end_time",
    "window_start_sec",
    "window_end_sec",
    "manual_decision",
    "manual_subtype_raw",
    "manual_subtype_norm",
    "review_bucket",
    "v3_cough_prob",
    "yamnet_top1",
    "yamnet_cough_score",
    "yamnet_speech_score",
    "yamnet_throat_like_score",
    "candidate_id",
    "clip_path",
    "review_clip_is_training_source",
    "canonical_training_window",
    "notes",
]
MINING_ROLE_ORDER = [
    "fp_hard_negative",
    "mid_negative",
    "clean_negative",
    "fn_positive",
    "confirmed_positive",
    "legacy_positive",
    "legacy_hard_negative",
    "legacy_public_non_cough",
]
SAMPLE_WEIGHT_BY_MINING_ROLE = {
    "fp_hard_negative": 8.0,
    "mid_negative": 4.0,
    "fn_positive": 5.0,
    "confirmed_positive": 3.0,
    "clean_negative": 1.0,
    "legacy_positive": 1.0,
    "legacy_hard_negative": 1.0,
    "legacy_public_non_cough": 1.0,
}
NOT_EXECUTED = [
    "one-epoch training",
    "formal training",
    "model export",
    "final label merge",
    "audio_labels_v3.csv overwrite",
    "raw-data modification",
    "OPERA/HeAR/YAMNet distillation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare v3.1 FP-fix train manifest/config and run mining_role preprocess smoke."
    )
    parser.add_argument("--base-labels", default="data/labels/audio_labels_v3.csv")
    parser.add_argument(
        "--addon-labels",
        default="review/board_live_data_1/audio_labels_v3_1_candidate_board_live_data_1.csv",
    )
    parser.add_argument("--out-labels", default="data/labels/audio_labels_v3_1_fpfix_train.csv")
    parser.add_argument("--out-sessions", default="data/labels/audio_labels_v3_1_fpfix_train_sessions.csv")
    parser.add_argument("--out-config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_merge_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_merge_summary.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preprocess-per-role", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode CSV: {path}")


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write the fpfix config") from exc

    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def check_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")


def non_empty(value: object) -> bool:
    return str(value).strip() != ""


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype(str).value_counts(dropna=False).sort_index().items()
    }


def split_distribution(frame: pd.DataFrame) -> dict[str, int]:
    return value_counts(frame, "split")


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> list[str]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return missing


def base_mining_role(row: pd.Series) -> str:
    label = str(row.get("label", "")).strip()
    if label == "cough":
        return "legacy_positive"
    if parse_bool(row.get("hard_negative", "")):
        return "legacy_hard_negative"
    return "legacy_public_non_cough"


def addon_mining_role(row: pd.Series) -> str:
    label = str(row["label"]).strip()
    cough_prob = float(row["v3_cough_prob"])
    if label == "cough" and cough_prob < 0.50:
        return "fn_positive"
    if label == "cough" and cough_prob >= 0.50:
        return "confirmed_positive"
    if label == "non_cough" and cough_prob >= 0.90:
        return "fp_hard_negative"
    if label == "non_cough" and cough_prob >= 0.50:
        return "mid_negative"
    if label == "non_cough" and cough_prob < 0.50:
        return "clean_negative"
    raise ValueError(f"Unsupported addon label/prob combination: label={label} v3_cough_prob={cough_prob}")


def first_audio_reference(row: pd.Series) -> tuple[str, str]:
    for column in ("canonical_audio_path", "source_audio", "original_path"):
        value = str(row.get(column, "")).strip()
        if value:
            return value, column
    raise ValueError(f"Addon row is missing canonical_audio_path/source_audio/original_path: {row.to_dict()}")


def ensure_positive_sample_weight(frame: pd.DataFrame) -> pd.Series:
    values = pd.to_numeric(frame.get("sample_weight", pd.Series([""] * len(frame))), errors="coerce")
    values = values.where(values > 0, 1.0).fillna(1.0)
    return values.astype(float)


def build_base_rows(base: pd.DataFrame) -> pd.DataFrame:
    output = base.copy()
    for column in OUTPUT_PRIORITY_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output["row_origin"] = "base_v3"
    output["mining_role"] = output.apply(base_mining_role, axis=1)
    output["sample_weight"] = ensure_positive_sample_weight(output)
    output["source"] = output["source"].where(output["source"].astype(str).str.strip().ne(""), "audio_labels_v3")
    output["source_start_time"] = output["source_start_time"].where(
        output["source_start_time"].astype(str).str.strip().ne(""),
        output["start_time"],
    )
    output["source_end_time"] = output["source_end_time"].where(
        output["source_end_time"].astype(str).str.strip().ne(""),
        output["end_time"],
    )
    return output


def build_addon_rows(addon: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    audio_resolution_counts = {"canonical_audio_path": 0, "source_audio": 0, "original_path": 0}

    for index, row in addon.iterrows():
        audio_file, audio_source_column = first_audio_reference(row)
        audio_resolution_counts[audio_source_column] += 1
        start = float(row["window_start_sec"])
        end = float(row["window_end_sec"])
        if end <= start:
            raise ValueError(f"Addon row {index} has end <= start: {start} -> {end}")
        mining_role = addon_mining_role(row)
        label = str(row["label"]).strip()
        negative_type = str(row.get("negative_type", "")).strip()
        hard_negative = label == "non_cough" and (
            negative_type == "hard_negative" or mining_role == "fp_hard_negative"
        )
        values: dict[str, object] = {column: "" for column in OUTPUT_PRIORITY_COLUMNS}
        for column in ADDON_TRACE_COLUMNS:
            values[column] = row.get(column, "")
        values.update(
            {
                "clip_id": str(row.get("clip_id", "")).strip() or str(row.get("candidate_id", "")).strip(),
                "session_id": row["session_id"],
                "person_id": row["person_id"],
                "start_time": start,
                "end_time": end,
                "label": label,
                "audio_file": audio_file,
                "split": "train",
                "hard_negative": bool(hard_negative),
                "sample_weight": SAMPLE_WEIGHT_BY_MINING_ROLE[mining_role],
                "source": "board_live_data_1_manual_review",
                "source_domain": "board_live_data_1",
                "change_reason": "v3_1_fpfix_board_live_addon",
                "hard_negative_source": f"board_live_data_1_{mining_role}" if label == "non_cough" else "",
                "negative_type": negative_type,
                "mining_role": mining_role,
                "row_origin": "board_live_data_1_addon",
                "canonical_audio_path": audio_file,
                "source_audio": str(row.get("source_audio", "")).strip() or audio_file,
                "source_start_time": start,
                "source_end_time": end,
                "candidate_id": row.get("candidate_id", ""),
                "clip_path": row.get("clip_path", ""),
                "review_clip_is_training_source": "False",
                "canonical_training_window": str(row.get("canonical_training_window", "")).strip()
                or "original_1s_window",
            }
        )
        rows.append(values)

    return pd.DataFrame(rows), audio_resolution_counts


def ordered_columns(*frames: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in OUTPUT_PRIORITY_COLUMNS:
        if column not in columns:
            columns.append(column)
    for frame in frames:
        for column in frame.columns:
            if column not in columns:
                columns.append(column)
    return columns


def build_sessions(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_id, group in merged.groupby("session_id", dropna=False):
        starts = pd.to_numeric(group["start_time"], errors="coerce")
        ends = pd.to_numeric(group["end_time"], errors="coerce")
        first = group.iloc[0]
        rows.append(
            {
                "session_id": session_id,
                "person_id": first.get("person_id", ""),
                "source": first.get("source", ""),
                "source_domain": first.get("source_domain", ""),
                "label": first.get("label", ""),
                "audio_file": first.get("audio_file", ""),
                "start_time": float(starts.min()) if starts.notna().any() else "",
                "end_time": float(ends.max()) if ends.notna().any() else "",
                "split": first.get("split", ""),
            }
        )
    return pd.DataFrame(rows)


def fpfix_config() -> dict[str, Any]:
    return {
        "project": {"name": "health_monitor_ai", "seed": 42},
        "paths": {
            "audio_dir": ".",
            "labels_csv": "data/labels/audio_labels_v3_1_fpfix_train.csv",
            "sessions_csv": "data/labels/audio_labels_v3_1_fpfix_train_sessions.csv",
            "processed_dir": "data/processed/audio_features_v3_1_fpfix",
            "models_dir": "models/audio",
            "results_dir": "results/audio",
        },
        "audio": BOARD_HTK_NO_NORM_V1,
        "labels": {
            "positive": "cough",
            "negative": "non_cough",
            "class_order": CLASS_ORDER,
        },
        "split": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "group_column": "person_id",
        },
        "sampling": {
            "mode": "balanced",
            "hard_negative_oversample": False,
            "hard_negative_multiplier": 1.0,
            "use_sample_weight": True,
            "sample_weight_plan": SAMPLE_WEIGHT_BY_MINING_ROLE,
            "note": "sample_weight generated for fpfix planning; train script support is reported separately.",
        },
        "training": {
            "model": "ds_cnn",
            "batch_size": 32,
            "epochs": 30,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_workers": 0,
            "device": "auto",
            "early_stopping_patience": 5,
            "optimizer": "adamw",
            "class_weight": "auto",
        },
        "eval": {
            "primary_metric": "val_f1",
            "threshold_min_recall": 0.90,
            "false_alarm": {"event_merge_gap_seconds": 5.0, "night_hours": 8.0},
        },
        "run": {"name": "audio_baseline_v3_1_fpfix"},
        "debug": {"small_data": False},
    }


def inspect_sample_weight_consumption() -> dict[str, Any]:
    sampling_path = ROOT / "src" / "audio" / "sampling.py"
    train_path = ROOT / "scripts" / "train_audio_baseline.py"
    sampling_text = sampling_path.read_text(encoding="utf-8") if sampling_path.exists() else ""
    train_text = train_path.read_text(encoding="utf-8") if train_path.exists() else ""
    sampling_module_supports = "sample_weight" in sampling_text and "sample_weights" in sampling_text
    train_consumes = any(
        token in train_text
        for token in (
            "build_weighted_sampler",
            "sample_weights(",
            "WeightedRandomSampler",
            "loss_class_weights",
        )
    )
    status = "active" if train_consumes else "generated_not_active"
    message = (
        "sample_weight generated and train_audio_baseline.py appears to consume it"
        if train_consumes
        else "sample_weight generated, training consumption not active/verified"
    )
    return {
        "sample_weight_column_generated": True,
        "sampling_module_supports_sample_weight": bool(sampling_module_supports),
        "train_audio_baseline_consumes_sample_weight": bool(train_consumes),
        "status": status,
        "message": message,
        "checked_files": [str(sampling_path), str(train_path)],
    }


def resolve_audio_path(audio_value: object) -> Path:
    value = Path(str(audio_value))
    if value.is_absolute():
        return value
    root_relative = ROOT / value
    if root_relative.exists():
        return root_relative
    return value


def run_preprocess_smoke(labels: pd.DataFrame, per_role: int, seed: int) -> dict[str, Any]:
    rng_seed = int(seed)
    selected_frames: list[pd.DataFrame] = []
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    warnings: list[str] = []

    for role_index, role in enumerate(MINING_ROLE_ORDER):
        role_rows = labels[labels["mining_role"].astype(str) == role].copy()
        available_counts[role] = int(len(role_rows))
        sample_count = min(int(per_role), len(role_rows))
        selected_counts[role] = int(sample_count)
        if sample_count == 0:
            warnings.append(f"No rows available for mining_role={role}")
            continue
        selected = role_rows.sample(n=sample_count, random_state=rng_seed + role_index).copy()
        selected_frames.append(selected)

    selected_rows = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    errors: list[str] = []
    checked_rows: list[dict[str, Any]] = []
    feature_values: list[np.ndarray] = []
    nan_inf_count = 0
    bad_shape_count = 0

    for _, row in selected_rows.iterrows():
        clip_id = str(row.get("clip_id", ""))
        role = str(row.get("mining_role", ""))
        audio_path = resolve_audio_path(row["audio_file"])
        try:
            start = float(row["start_time"])
            end = float(row["end_time"])
            smoke_end = min(end, start + float(BOARD_HTK_NO_NORM_V1["window_seconds"]))
            if smoke_end <= start:
                smoke_end = end
            segment = load_audio_segment(
                audio_path,
                sample_rate=int(BOARD_HTK_NO_NORM_V1["sample_rate"]),
                start_time=start,
                end_time=smoke_end,
            )
            windows = iter_windows(
                segment,
                sample_rate=int(BOARD_HTK_NO_NORM_V1["sample_rate"]),
                window_seconds=float(BOARD_HTK_NO_NORM_V1["window_seconds"]),
                hop_seconds=float(BOARD_HTK_NO_NORM_V1["hop_seconds"]),
            )
            if not windows:
                errors.append(f"No feature window produced for clip_id={clip_id} role={role}")
                continue
            feature = log_mel_feature(windows[0][2], BOARD_HTK_NO_NORM_V1)
        except Exception as exc:
            errors.append(f"Preprocess smoke failed clip_id={clip_id} role={role}: {exc}")
            continue

        if tuple(feature.shape) != EXPECTED_FEATURE_SHAPE:
            bad_shape_count += 1
            errors.append(f"Bad feature shape clip_id={clip_id}: {list(feature.shape)}")
        if not np.isfinite(feature).all():
            nan_inf_count += 1
            errors.append(f"NaN/Inf feature values clip_id={clip_id}")
        feature_values.append(feature)
        checked_rows.append(
            {
                "clip_id": clip_id,
                "mining_role": role,
                "label": str(row.get("label", "")),
                "person_id": str(row.get("person_id", "")),
                "audio_file": str(row.get("audio_file", "")),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "feature_shape": [int(value) for value in feature.shape],
            }
        )

    stats: dict[str, Any] = {"count": 0}
    if feature_values:
        stack = np.stack(feature_values)
        stats = {
            "count": int(stack.shape[0]),
            "shape": [int(value) for value in feature_values[0].shape],
            "min": float(stack.min()),
            "mean": float(stack.mean()),
            "std": float(stack.std()),
            "max": float(stack.max()),
        }

    return {
        "status": "passed" if not errors else "failed",
        "seed": rng_seed,
        "per_role": int(per_role),
        "roles_requested": MINING_ROLE_ORDER,
        "available_counts": available_counts,
        "selected_counts": selected_counts,
        "selected_rows": int(len(selected_rows)),
        "checked_rows": int(len(checked_rows)),
        "audio_frontend": "board_htk_no_norm_v1",
        "audio_mix_mode": "librosa_mono_average",
        "expected_feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "bad_shape_feature_count": int(bad_shape_count),
        "nan_inf_feature_count": int(nan_inf_count),
        "feature_stats": stats,
        "checked_rows_preview": checked_rows[:30],
        "warnings": warnings,
        "errors": errors,
    }


def validate_frontend(config: dict[str, Any]) -> dict[str, Any]:
    audio = config["audio"]
    checks = {
        "frontend": audio.get("frontend") == "board_htk_no_norm_v1",
        "sample_rate": int(audio.get("sample_rate")) == 16000,
        "window_seconds": float(audio.get("window_seconds")) == 1.0,
        "hop_seconds": float(audio.get("hop_seconds")) == 0.5,
        "n_mels": int(audio.get("n_mels")) == 40,
        "mel_scale_htk": str(audio.get("mel_scale")).lower() == "htk" and bool(audio.get("htk")) is True,
        "normalize_none": str(audio.get("normalize")).lower() == "none",
        "class_order": config["labels"].get("class_order") == CLASS_ORDER,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def markdown_table(items: dict[str, int | float | str]) -> list[str]:
    if not items:
        return ["_None_"]
    lines = ["| key | value |", "| --- | --- |"]
    for key, value in items.items():
        lines.append(f"| {key} | {value} |")
    return lines


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3.1 FP-fix Prep Summary",
        "",
        f"- status: `{report['status']}`",
        f"- base_rows: `{report['rows']['base_v3']}`",
        f"- addon_rows: `{report['rows']['board_live_data_1_addon']}`",
        f"- combined_rows: `{report['rows']['combined']}`",
        f"- base_split_distribution_unchanged: `{report['base_split_distribution_unchanged']}`",
        f"- sample_weight_status: `{report['training_sample_weight_support']['message']}`",
        f"- preprocess_smoke: `{report['preprocess_smoke']['status']}`",
        "",
        "## Mining Role Counts",
        "",
    ]
    lines.extend(markdown_table(report["mining_role_counts"]))
    lines.extend(["", "## Addon Label Counts", ""])
    lines.extend(markdown_table(report["addon_label_counts"]))
    lines.extend(["", "## Audio Path Resolution", ""])
    lines.extend(markdown_table(report["audio_path_resolution"]))
    lines.extend(["", "## Preprocess Smoke", ""])
    lines.append(f"- selected_rows: `{report['preprocess_smoke']['selected_rows']}`")
    lines.append(f"- checked_rows: `{report['preprocess_smoke']['checked_rows']}`")
    lines.append(f"- selected_counts: `{report['preprocess_smoke']['selected_counts']}`")
    lines.append(f"- warnings: `{report['preprocess_smoke']['warnings']}`")
    lines.append(f"- errors: `{report['preprocess_smoke']['errors']}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- board_live_data_1 addon is p01-only and is for train addon or regression check only.",
            "- It must not be used as a standalone held-out generalization evaluation set.",
            "- Review clips are trace/listening aids; canonical training windows use original source audio plus start/end.",
            "",
            "## Not Executed",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in report["not_executed"]])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_path = resolve_path(args.base_labels)
    addon_path = resolve_path(args.addon_labels)
    out_labels_path = resolve_path(args.out_labels)
    out_sessions_path = resolve_path(args.out_sessions)
    out_config_path = resolve_path(args.out_config)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    for path in (out_labels_path, out_sessions_path, out_config_path, report_path, summary_path):
        check_output_path(path, args.overwrite)
    if not base_path.exists():
        raise FileNotFoundError(f"Base labels not found: {base_path}")
    if not addon_path.exists():
        raise FileNotFoundError(f"Addon labels not found: {addon_path}")

    base = read_csv(base_path)
    addon = read_csv(addon_path)
    require_columns(base, base_path, BASE_REQUIRED_COLUMNS)
    require_columns(addon, addon_path, ADDON_REQUIRED_COLUMNS)

    addon_missing_trace = {
        column: int(addon[column].astype(str).str.strip().eq("").sum())
        for column in ("original_path", "source_audio", "window_start_sec", "window_end_sec")
    }
    missing_required = {key: value for key, value in addon_missing_trace.items() if value}
    if missing_required:
        raise ValueError(f"Addon required trace fields are empty: {missing_required}")

    base_split_before = split_distribution(base)
    base_rows = build_base_rows(base)
    addon_rows, audio_resolution_counts = build_addon_rows(addon)

    columns = ordered_columns(base_rows, addon_rows)
    for column in columns:
        if column not in base_rows.columns:
            base_rows[column] = ""
        if column not in addon_rows.columns:
            addon_rows[column] = ""
    merged = pd.concat([base_rows[columns], addon_rows[columns]], ignore_index=True)
    sessions = build_sessions(merged)
    config = fpfix_config()

    base_after = merged[merged["row_origin"].astype(str) == "base_v3"].copy()
    base_split_after = split_distribution(base_after)
    split_unchanged = base_split_before == base_split_after
    preprocess_smoke = run_preprocess_smoke(merged, per_role=args.preprocess_per_role, seed=args.seed)
    frontend_check = validate_frontend(config)
    sample_weight_support = inspect_sample_weight_consumption()

    errors: list[str] = []
    if not split_unchanged:
        errors.append("Base v3 split distribution changed after merge")
    if int(len(base_after)) != int(len(base)):
        errors.append("Base v3 row count changed after merge")
    if int(len(addon_rows)) != int(len(addon)):
        errors.append("Addon row count changed after merge")
    if merged["mining_role"].astype(str).str.strip().eq("").any():
        errors.append("mining_role contains empty values")
    if frontend_check["status"] != "passed":
        errors.append("board_htk_no_norm_v1 frontend check failed")
    if preprocess_smoke["status"] != "passed":
        errors.append("preprocess smoke failed")

    report = {
        "status": "passed" if not errors else "failed",
        "paths": {
            "base_labels": str(base_path),
            "addon_labels": str(addon_path),
            "out_labels": str(out_labels_path),
            "out_sessions": str(out_sessions_path),
            "out_config": str(out_config_path),
            "report": str(report_path),
            "summary_md": str(summary_path),
        },
        "rows": {
            "base_v3": int(len(base)),
            "board_live_data_1_addon": int(len(addon_rows)),
            "combined": int(len(merged)),
        },
        "label_counts": value_counts(merged, "label"),
        "addon_label_counts": value_counts(addon_rows, "label"),
        "addon_negative_type_counts": value_counts(addon_rows, "negative_type"),
        "mining_role_counts": value_counts(merged, "mining_role"),
        "sample_weight_by_mining_role": SAMPLE_WEIGHT_BY_MINING_ROLE,
        "training_sample_weight_support": sample_weight_support,
        "base_split_distribution_before": base_split_before,
        "base_split_distribution_after": base_split_after,
        "base_split_distribution_unchanged": bool(split_unchanged),
        "split_counts": value_counts(merged, "split"),
        "audio_path_resolution": audio_resolution_counts,
        "addon_required_trace_empty_counts": addon_missing_trace,
        "person_id_counts": value_counts(merged, "person_id"),
        "addon_person_id_counts": value_counts(addon_rows, "person_id"),
        "p01_policy": {
            "all_addon_rows_p01": set(addon_rows["person_id"].astype(str)) == {"p01"},
            "usage": "train addon or regression check only; not a standalone held-out generalization evaluation set",
        },
        "canonical_training_window": {
            "policy": "original source audio plus window_start_sec/window_end_sec",
            "review_clip_is_training_source_true_rows": int(
                addon_rows["review_clip_is_training_source"].astype(str).str.lower().eq("true").sum()
            ),
        },
        "frontend_check": frontend_check,
        "preprocess_smoke": preprocess_smoke,
        "not_executed": NOT_EXECUTED,
        "errors": errors,
    }

    write_json(report, report_path)
    write_summary(report, summary_path)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors, "report": str(report_path)}, indent=2))
        return 1

    ensure_dir(out_labels_path.parent)
    ensure_dir(out_sessions_path.parent)
    merged.to_csv(out_labels_path, index=False, encoding="utf-8")
    sessions.to_csv(out_sessions_path, index=False, encoding="utf-8")
    write_yaml(config, out_config_path)
    print(
        json.dumps(
            {
                "status": "passed",
                "out_labels": str(out_labels_path),
                "out_config": str(out_config_path),
                "rows": report["rows"],
                "preprocess_smoke": preprocess_smoke["status"],
                "sample_weight": sample_weight_support["message"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
