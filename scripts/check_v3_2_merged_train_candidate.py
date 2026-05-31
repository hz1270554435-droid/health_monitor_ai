from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import iter_windows, load_audio_segment, log_mel_feature  # noqa: E402


EXPECTED_FEATURE_SHAPE = (40, 101)
OLD_LABEL_PATHS = [
    "data/labels/audio_labels_v3.csv",
    "data/labels/audio_labels_v3_1_fpfix_train.csv",
]
FORBIDDEN_TRAIN_LABEL_PATHS = [
    "data/labels/audio_labels_v3_2_train.csv",
    "review/v3_2_recall_fix/audio_labels_v3_2_train.csv",
]
REQUIRED_FIELDS = [
    "sample_id",
    "clip_id",
    "label",
    "source_stage",
    "source_audio_path",
    "source_start_time",
    "source_end_time",
    "split",
    "source_domain",
]
ADDON_REQUIRED_FIELDS = [
    "review_id",
    "candidate_role",
    "downstream_decision",
    "person_id_for_analysis",
    "source_actor_type",
    "scenario_canonical",
    "split_role_suggested",
    "proposed_window_start_sec",
    "proposed_window_end_sec",
    "fpfix_cough_prob",
    "merge_policy_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only dataset check and board_htk_no_norm_v1 preprocess smoke for v3.2 merged train candidate labels."
    )
    parser.add_argument(
        "--labels",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_merged_candidate.csv",
    )
    parser.add_argument(
        "--merge-report",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_merged_candidate_report.json",
    )
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument(
        "--dataset-report",
        default="review/v3_2_recall_fix/v3_2_merged_train_candidate_dataset_check_report.json",
    )
    parser.add_argument(
        "--dataset-summary",
        default="review/v3_2_recall_fix/v3_2_merged_train_candidate_dataset_check_summary.md",
    )
    parser.add_argument(
        "--preprocess-report",
        default="review/v3_2_recall_fix/v3_2_merged_train_candidate_preprocess_smoke_report.json",
    )
    parser.add_argument(
        "--preprocess-summary",
        default="review/v3_2_recall_fix/v3_2_merged_train_candidate_preprocess_smoke_summary.md",
    )
    parser.add_argument(
        "--allow-final-label",
        action="store_true",
        help="Allow the checked labels path itself to be data/labels/audio_labels_v3_2_train.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_new_file(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def sha256_or_missing(path: Path) -> str:
    if not path.exists():
        return "<missing>"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependency check
        raise RuntimeError("PyYAML is required to read the audio config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def crosstab(rows: list[dict[str, str]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[row.get(left, "")][row.get(right, "")] += 1
    return {key: dict(value.most_common()) for key, value in sorted(table.items())}


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def first_nonempty(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def row_audio_path(row: dict[str, str]) -> Path:
    return resolve_path(first_nonempty(row, ["source_audio_path", "canonical_audio_path", "source_audio", "original_path", "audio_file"]))


def row_start_end(row: dict[str, str]) -> tuple[float, float]:
    start = parse_float(first_nonempty(row, ["source_start_time", "window_start_sec", "proposed_window_start_sec", "start_time"]))
    end = parse_float(first_nonempty(row, ["source_end_time", "window_end_sec", "proposed_window_end_sec", "end_time"]))
    return start, end


def wav_info(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as f:
        sample_rate = int(f.getframerate())
        channels = int(f.getnchannels())
        frames = int(f.getnframes())
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_s": frames / float(sample_rate) if sample_rate else 0.0,
    }


def validate_audio_config(config: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "frontend_board_htk_no_norm_v1": config.get("frontend") == "board_htk_no_norm_v1",
        "sample_rate_16000": int(config.get("sample_rate", 0)) == 16000,
        "window_seconds_1_0": float(config.get("window_seconds", 0)) == 1.0,
        "hop_seconds_0_5": float(config.get("hop_seconds", 0)) == 0.5,
        "n_mels_40": int(config.get("n_mels", 0)) == 40,
        "n_fft_1024": int(config.get("n_fft", 0)) == 1024,
        "hop_length_160": int(config.get("hop_length", 0)) == 160,
        "htk_true": bool(config.get("htk", False)) is True,
        "mel_scale_htk": str(config.get("mel_scale", "")).lower() == "htk",
        "normalize_none": str(config.get("normalize", "")).lower() == "none",
    }
    return {
        "audio_frontend": config.get("frontend"),
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }


def feature_stats(features: list[np.ndarray]) -> dict[str, Any]:
    if not features:
        return {"count": 0}
    stacked = np.stack(features)
    return {
        "count": int(stacked.shape[0]),
        "shape": [int(value) for value in features[0].shape],
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "mean": float(stacked.mean()),
        "std": float(stacked.std()),
    }


def feature_stats_by(feature_rows: list[tuple[dict[str, str], np.ndarray]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, feature in feature_rows:
        grouped[row.get(field, "")].append(feature)
    return {key: feature_stats(values) for key, values in sorted(grouped.items())}


def compute_expected_feature_window_count(rows: list[dict[str, str]], sample_rate: int, window_seconds: float, hop_seconds: float) -> int:
    total = 0
    window_size = int(round(sample_rate * window_seconds))
    hop_size = int(round(sample_rate * hop_seconds))
    for row in rows:
        start, end = row_start_end(row)
        if math.isnan(start) or math.isnan(end) or end <= start:
            continue
        segment_size = int(round((end - start) * sample_rate))
        if segment_size <= 0:
            continue
        max_start = max(0, segment_size - window_size)
        total += len(list(range(0, max_start + 1, hop_size))) or 1
    return total


def run_dataset_check(
    rows: list[dict[str, str]],
    columns: list[str],
    merge_report: dict[str, Any],
    labels_path: Path,
    allow_final_label: bool,
    old_hash_before: dict[str, str],
    old_hash_after: dict[str, str],
    labels_hash_before: str,
    labels_hash_after: str,
) -> dict[str, Any]:
    expected_rows = int(merge_report.get("row_counts", {}).get("merged_rows", len(rows)))
    expected_labels = {
        str(key): int(value)
        for key, value in merge_report.get("label_distribution", {}).get("merged", counts([row.get("label", "") for row in rows])).items()
    }
    expected_source_stage = {
        str(key): int(value)
        for key, value in merge_report.get("source_stage_distribution", counts([row.get("source_stage", "") for row in rows])).items()
    }
    required_missing = [field for field in REQUIRED_FIELDS if field not in columns]
    addon_missing = [
        field
        for field in ADDON_REQUIRED_FIELDS
        if any(row.get("source_stage") == "v3_2_recall_fix_addon" for row in rows) and field not in columns
    ]
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_root = ROOT / "data" / "raw"
    audio_cache: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    window_durations: list[float] = []
    sample_rate_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    channels_by_stage: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        sample_id = row.get("sample_id", "")
        stage = row.get("source_stage", "")
        audio_path = row_audio_path(row)
        if is_under(audio_path, raw_root):
            failures.append({"sample_id": sample_id, "kind": "ml_data_raw_path", "path": str(audio_path)})
        if not audio_path.exists():
            failures.append({"sample_id": sample_id, "kind": "missing_audio", "path": str(audio_path)})
            continue
        key = str(audio_path)
        if key not in audio_cache:
            try:
                audio_cache[key] = wav_info(audio_path)
            except Exception as exc:  # pragma: no cover - defensive audit path
                audio_cache[key] = {"error": str(exc)}
        info = audio_cache[key]
        if "error" in info:
            failures.append({"sample_id": sample_id, "kind": "wav_header_error", "error": info["error"]})
            continue
        sample_rate_by_stage[stage][str(info["sample_rate"])] += 1
        channels_by_stage[stage][str(info["channels"])] += 1
        start, end = row_start_end(row)
        duration = end - start if not math.isnan(start) and not math.isnan(end) else math.nan
        if not math.isnan(duration):
            window_durations.append(duration)
        if math.isnan(start) or math.isnan(end) or start < 0 or end <= start:
            failures.append({"sample_id": sample_id, "kind": "bad_time_bounds", "start": start, "end": end})
        elif end > float(info["duration_s"]) + 1e-6:
            failures.append(
                {
                    "sample_id": sample_id,
                    "kind": "time_bounds_exceed_wav_duration",
                    "start": start,
                    "end": end,
                    "duration_s": info["duration_s"],
                }
            )

    if required_missing:
        errors.append({"kind": "missing_required_fields", "fields": required_missing})
    if addon_missing:
        errors.append({"kind": "missing_addon_required_fields", "fields": addon_missing})
    labels = [row.get("label", "") for row in rows]
    sample_ids = [row.get("sample_id", "") for row in rows]
    source_stages = [row.get("source_stage", "") for row in rows]
    invalid_labels = sorted(set(labels) - {"cough", "non_cough"})
    if invalid_labels:
        errors.append({"kind": "invalid_labels", "values": invalid_labels})
    if len(sample_ids) != len(set(sample_ids)):
        errors.append({"kind": "duplicate_sample_id", "count": len(sample_ids) - len(set(sample_ids))})
    if failures:
        errors.append({"kind": "path_or_boundary_failures", "count": len(failures), "examples": failures[:30]})

    train_paths = [resolve_path(path) for path in FORBIDDEN_TRAIN_LABEL_PATHS]
    allowed_final_label = resolve_path(FORBIDDEN_TRAIN_LABEL_PATHS[0]).resolve()
    checked_labels_resolved = labels_path.resolve()
    train_outputs_found = [
        str(path)
        for path in train_paths
        if path.exists() and not (allow_final_label and path.resolve() == checked_labels_resolved == allowed_final_label)
    ]
    if train_outputs_found:
        errors.append({"kind": "forbidden_train_label_exists", "paths": train_outputs_found})

    addon_rows = [row for row in rows if row.get("source_stage") == "v3_2_recall_fix_addon"]
    addon_forbidden_counts = {
        "p03": sum(1 for row in addon_rows if row.get("person_id_for_analysis") == "p03"),
        "p02": sum(1 for row in addon_rows if row.get("person_id_for_analysis") == "p02"),
        "long_quiet_eval": sum(1 for row in addon_rows if row.get("split_role_suggested") == "long_quiet_eval"),
        "distance_extrap_eval": sum(1 for row in addon_rows if row.get("split_role_suggested") == "distance_extrap_eval"),
    }
    sample_rate_counts = Counter()
    channel_counts = Counter()
    for info in audio_cache.values():
        if "sample_rate" in info:
            sample_rate_counts[str(info["sample_rate"])] += 1
        if "channels" in info:
            channel_counts[str(info["channels"])] += 1

    strict_channels_all_1 = set(channel_counts.keys()).issubset({"1"}) and bool(channel_counts)
    addon_channels_all_1 = channels_by_stage.get("v3_2_recall_fix_addon") == {"1": len(addon_rows)}
    legacy_base_channel_caveat = (
        not strict_channels_all_1
        and addon_channels_all_1
        and "2" in channels_by_stage.get("v3_1_fpfix_base", {})
    )
    if legacy_base_channel_caveat:
        warnings.append("Preserved v3.1-fpfix base contains existing 2-channel board_live_data_1 WAV rows; preprocess uses mono=True.")

    validation = {
        "row_count_matches_merge_report": len(rows) == expected_rows,
        "labels_only_cough_non_cough": set(labels).issubset({"cough", "non_cough"}),
        "label_distribution_matches_merge_report": counts(labels) == expected_labels,
        "source_stage_distribution_matches_merge_report": counts(source_stages) == expected_source_stage,
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)),
        "source_audio_paths_exist_and_bounds_ok": len(failures) == 0,
        "sample_rate_all_16000": set(sample_rate_counts.keys()).issubset({"16000"}) and bool(sample_rate_counts),
        "addon_channels_all_1": addon_channels_all_1,
        "strict_merged_channels_all_1": strict_channels_all_1,
        "p03_p02_long_quiet_distance_not_in_v3_2_addon": not any(addon_forbidden_counts.values()),
        "old_label_hashes_unchanged": old_hash_before == old_hash_after,
        "merged_candidate_csv_unchanged": labels_hash_before == labels_hash_after,
        "no_unexpected_final_audio_labels_v3_2_train": not train_outputs_found,
        "checked_final_label_allowed": (not allow_final_label) or checked_labels_resolved == allowed_final_label,
        "ml_data_raw_not_written_or_referenced": not any(item.get("kind") == "ml_data_raw_path" for item in failures),
        "training_not_run": True,
    }
    critical_validation_keys = [key for key in validation if key != "strict_merged_channels_all_1"]
    critical_pass = all(validation[key] for key in critical_validation_keys) and not errors
    if critical_pass and validation["strict_merged_channels_all_1"]:
        result = "pass"
    elif critical_pass and legacy_base_channel_caveat:
        result = "pass_with_legacy_base_channel_caveat"
    else:
        result = "fail"

    return {
        "stage": "v3.2-CHECK-2-DATASET",
        "checked_rows": len(rows),
        "expected_rows": expected_rows,
        "unique_audio_paths": len(audio_cache),
        "missing_required_fields": required_missing,
        "missing_addon_required_fields": addon_missing,
        "distribution": {
            "label_distribution": counts(labels),
            "source_stage_distribution": counts(source_stages),
            "candidate_role_distribution": counts([row.get("candidate_role", "") for row in rows]),
            "candidate_role_distribution_addon": counts([row.get("candidate_role", "") for row in addon_rows]),
            "person_distribution_addon": counts([row.get("person_id_for_analysis", "") for row in addon_rows]),
            "scenario_distribution_addon": counts([row.get("scenario_canonical", "") for row in addon_rows]),
            "source_actor_type_distribution_addon": counts([row.get("source_actor_type", "") for row in addon_rows]),
            "candidate_role_by_label_addon": crosstab(addon_rows, "candidate_role", "label"),
        },
        "sample_rate_counts": dict(sample_rate_counts.most_common()),
        "channels_counts": dict(channel_counts.most_common()),
        "sample_rate_counts_by_stage": {key: dict(value.most_common()) for key, value in sorted(sample_rate_by_stage.items())},
        "channels_counts_by_stage": {key: dict(value.most_common()) for key, value in sorted(channels_by_stage.items())},
        "legacy_base_channel_caveat": legacy_base_channel_caveat,
        "window_duration_summary": {
            "count": len(window_durations),
            "min": min(window_durations) if window_durations else None,
            "max": max(window_durations) if window_durations else None,
        },
        "addon_forbidden_holdout_counts": addon_forbidden_counts,
        "train_outputs_found": train_outputs_found,
        "allow_final_label": allow_final_label,
        "checked_labels_path": str(labels_path),
        "failures": failures[:50],
        "failure_count": len(failures),
        "failures_truncated": len(failures) > 50,
        "errors": errors,
        "warnings": warnings,
        "validation": validation,
        "result": result,
    }


def run_preprocess_smoke(
    rows: list[dict[str, str]],
    audio_config: dict[str, Any],
    frontend_check: dict[str, Any],
    old_hash_before: dict[str, str],
    old_hash_after: dict[str, str],
    labels_hash_before: str,
    labels_hash_after: str,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    feature_rows: list[tuple[dict[str, str], np.ndarray]] = []
    failed_rows = 0
    failed_windows = 0
    shape_counts: Counter[str] = Counter()
    bad_shape_count = 0
    nan_inf_count = 0
    generated_window_counts_by_stage: Counter[str] = Counter()
    generated_window_counts_by_label: Counter[str] = Counter()

    for row in rows:
        sample_id = row.get("sample_id", "")
        try:
            start, end = row_start_end(row)
            segment = load_audio_segment(
                row_audio_path(row),
                sample_rate=int(audio_config["sample_rate"]),
                start_time=start,
                end_time=end,
            )
            windows = iter_windows(
                segment,
                sample_rate=int(audio_config["sample_rate"]),
                window_seconds=float(audio_config["window_seconds"]),
                hop_seconds=float(audio_config["hop_seconds"]),
            )
            if not windows:
                raise ValueError("no windows produced")
        except Exception as exc:
            failed_rows += 1
            errors.append({"sample_id": sample_id, "kind": "row_feature_extract_failed", "error": str(exc)})
            continue

        for _, _, chunk in windows:
            try:
                feature = log_mel_feature(chunk, audio_config)
            except Exception as exc:
                failed_windows += 1
                errors.append({"sample_id": sample_id, "kind": "window_feature_extract_failed", "error": str(exc)})
                continue
            shape = tuple(int(value) for value in feature.shape)
            shape_counts[str(list(shape))] += 1
            if shape != EXPECTED_FEATURE_SHAPE:
                bad_shape_count += 1
                errors.append({"sample_id": sample_id, "kind": "bad_feature_shape", "shape": list(shape)})
            if not np.isfinite(feature).all():
                nan_inf_count += 1
                errors.append({"sample_id": sample_id, "kind": "nan_inf_feature"})
            feature_rows.append((row, feature))
            generated_window_counts_by_stage[row.get("source_stage", "")] += 1
            generated_window_counts_by_label[row.get("label", "")] += 1

    expected_feature_windows = compute_expected_feature_window_count(
        rows,
        sample_rate=int(audio_config["sample_rate"]),
        window_seconds=float(audio_config["window_seconds"]),
        hop_seconds=float(audio_config["hop_seconds"]),
    )
    validation = {
        "frontend_config_pass": frontend_check["result"] == "pass",
        "checked_label_rows_match_input": len(rows) > 0,
        "feature_success_windows_match_expected": len(feature_rows) == expected_feature_windows,
        "failed_label_rows_is_0": failed_rows == 0,
        "failed_feature_windows_is_0": failed_windows == 0,
        "feature_shape_all_40_101": dict(shape_counts) == {str([40, 101]): expected_feature_windows},
        "nan_inf_feature_count_is_0": nan_inf_count == 0,
        "bad_shape_feature_count_is_0": bad_shape_count == 0,
        "processed_feature_files_not_saved": True,
        "old_label_hashes_unchanged": old_hash_before == old_hash_after,
        "merged_candidate_csv_unchanged": labels_hash_before == labels_hash_after,
        "training_not_run": True,
        "ml_data_raw_not_written": True,
    }
    return {
        "stage": "v3.2-CHECK-2-PREPROCESS-SMOKE",
        "audio_frontend": "board_htk_no_norm_v1",
        "audio_config": audio_config,
        "frontend_check": frontend_check,
        "checked_label_rows": len(rows),
        "expected_feature_windows": expected_feature_windows,
        "feature_success_windows": len(feature_rows),
        "failed_label_rows": failed_rows,
        "failed_feature_windows": failed_windows,
        "expected_feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "feature_shape_counts": dict(shape_counts),
        "nan_inf_feature_count": nan_inf_count,
        "bad_shape_feature_count": bad_shape_count,
        "feature_window_counts_by_source_stage": dict(generated_window_counts_by_stage.most_common()),
        "feature_window_counts_by_label": dict(generated_window_counts_by_label.most_common()),
        "feature_stats": feature_stats([feature for _, feature in feature_rows]),
        "feature_stats_by_label": feature_stats_by(feature_rows, "label"),
        "feature_stats_by_source_stage": feature_stats_by(feature_rows, "source_stage"),
        "errors": errors[:50],
        "errors_truncated": len(errors) > 50,
        "validation": validation,
        "result": "pass" if all(validation.values()) and not errors else "fail",
    }


def write_dataset_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Merged Train-Candidate Dataset Check Summary",
        "",
        f"- result: `{report['result']}`",
        f"- checked_rows: `{report['checked_rows']}`",
        f"- unique_audio_paths: `{report['unique_audio_paths']}`",
        f"- legacy_base_channel_caveat: `{report['legacy_base_channel_caveat']}`",
        f"- failure_count: `{report['failure_count']}`",
        "",
        "## Label Distribution",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for key, value in report["distribution"]["label_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Source Stage Distribution", "", "| source_stage | count |", "| --- | ---: |"])
    for key, value in report["distribution"]["source_stage_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Validation", "", "| check | result |", "| --- | --- |"])
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preprocess_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Merged Train-Candidate Preprocess Smoke Summary",
        "",
        f"- result: `{report['result']}`",
        f"- audio_frontend: `{report['audio_frontend']}`",
        f"- checked_label_rows: `{report['checked_label_rows']}`",
        f"- expected_feature_windows: `{report['expected_feature_windows']}`",
        f"- feature_success_windows: `{report['feature_success_windows']}`",
        f"- failed_label_rows: `{report['failed_label_rows']}`",
        f"- failed_feature_windows: `{report['failed_feature_windows']}`",
        f"- feature_shape_counts: `{report['feature_shape_counts']}`",
        "",
        "## Feature Windows By Source Stage",
        "",
        "| source_stage | windows |",
        "| --- | ---: |",
    ]
    for key, value in report["feature_window_counts_by_source_stage"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Feature Stats", "", f"`{report['feature_stats']}`", "", "## Validation", "", "| check | result |", "| --- | --- |"])
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels_path = resolve_path(args.labels)
    merge_report_path = resolve_path(args.merge_report)
    config_path = resolve_path(args.config)
    dataset_report_path = resolve_path(args.dataset_report)
    dataset_summary_path = resolve_path(args.dataset_summary)
    preprocess_report_path = resolve_path(args.preprocess_report)
    preprocess_summary_path = resolve_path(args.preprocess_summary)

    for required in (labels_path, merge_report_path, config_path):
        if not required.exists():
            raise FileNotFoundError(required)
    for output in (dataset_report_path, dataset_summary_path, preprocess_report_path, preprocess_summary_path):
        ensure_new_file(output, args.overwrite)

    old_hash_before = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    labels_hash_before = sha256_or_missing(labels_path)
    rows, columns = read_csv(labels_path)
    merge_report = read_json(merge_report_path)
    config = read_yaml(config_path)
    audio_config = dict(config["audio"])
    frontend_check = validate_audio_config(audio_config)

    dataset_report = run_dataset_check(
        rows,
        columns,
        merge_report,
        labels_path,
        args.allow_final_label,
        old_hash_before,
        {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS},
        labels_hash_before,
        sha256_or_missing(labels_path),
    )
    preprocess_report = run_preprocess_smoke(
        rows,
        audio_config,
        frontend_check,
        old_hash_before,
        {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS},
        labels_hash_before,
        sha256_or_missing(labels_path),
    )

    old_hash_after = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    labels_hash_after = sha256_or_missing(labels_path)
    dataset_report["old_label_hash_after"] = old_hash_after
    dataset_report["merged_candidate_hash_after"] = labels_hash_after
    dataset_report["validation"]["old_label_hashes_unchanged"] = old_hash_before == old_hash_after
    dataset_report["validation"]["merged_candidate_csv_unchanged"] = labels_hash_before == labels_hash_after
    critical_keys = [key for key in dataset_report["validation"] if key != "strict_merged_channels_all_1"]
    critical_pass = all(dataset_report["validation"][key] for key in critical_keys) and not dataset_report["errors"]
    if critical_pass and dataset_report["validation"]["strict_merged_channels_all_1"]:
        dataset_report["result"] = "pass"
    elif critical_pass and dataset_report["legacy_base_channel_caveat"]:
        dataset_report["result"] = "pass_with_legacy_base_channel_caveat"
    else:
        dataset_report["result"] = "fail"

    preprocess_report["validation"]["old_label_hashes_unchanged"] = old_hash_before == old_hash_after
    preprocess_report["validation"]["merged_candidate_csv_unchanged"] = labels_hash_before == labels_hash_after
    preprocess_report["validation"]["training_not_run"] = True
    preprocess_report["validation"]["ml_data_raw_not_written"] = True
    preprocess_report["result"] = (
        "pass" if all(preprocess_report["validation"].values()) and not preprocess_report["errors"] else "fail"
    )

    dataset_report_path.write_text(json.dumps(dataset_report, indent=2, ensure_ascii=False), encoding="utf-8")
    preprocess_report_path.write_text(json.dumps(preprocess_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_dataset_summary(dataset_summary_path, dataset_report)
    write_preprocess_summary(preprocess_summary_path, preprocess_report)
    print(
        json.dumps(
            {
                "dataset_check_result": dataset_report["result"],
                "preprocess_smoke_result": preprocess_report["result"],
                "checked_rows": len(rows),
                "feature_success_windows": preprocess_report["feature_success_windows"],
                "failed_label_rows": preprocess_report["failed_label_rows"],
                "failed_feature_windows": preprocess_report["failed_feature_windows"],
                "feature_shape_counts": preprocess_report["feature_shape_counts"],
                "generated_files": [
                    str(dataset_report_path),
                    str(dataset_summary_path),
                    str(preprocess_report_path),
                    str(preprocess_summary_path),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
