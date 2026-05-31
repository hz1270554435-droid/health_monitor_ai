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
REQUIRED_FIELDS = [
    "sample_id",
    "review_id",
    "source_audio_path",
    "source_json_path",
    "label",
    "candidate_role",
    "downstream_decision",
    "person_id_for_analysis",
    "source_actor_type",
    "scenario_canonical",
    "split_role_suggested",
    "source_domain",
    "proposed_window_start_sec",
    "proposed_window_end_sec",
    "event_start_sec",
    "event_end_sec",
    "fpfix_cough_prob",
    "merge_policy_version",
    "notes",
]
OLD_LABEL_PATHS = [
    "data/labels/audio_labels_v3.csv",
    "data/labels/audio_labels_v3_1_fpfix_train.csv",
]
FORBIDDEN_TRAIN_LABEL_PATHS = [
    "data/labels/audio_labels_v3_2_train.csv",
    "review/v3_2_recall_fix/audio_labels_v3_2_train.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only dataset check and board_htk_no_norm_v1 preprocess smoke for v3.2 train-candidate labels."
    )
    parser.add_argument(
        "--labels",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run.csv",
    )
    parser.add_argument(
        "--merge-report",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run_report.json",
    )
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument(
        "--dataset-report",
        default="review/v3_2_recall_fix/v3_2_train_candidate_dataset_check_report.json",
    )
    parser.add_argument(
        "--dataset-summary",
        default="review/v3_2_recall_fix/v3_2_train_candidate_dataset_check_summary.md",
    )
    parser.add_argument(
        "--preprocess-report",
        default="review/v3_2_recall_fix/v3_2_train_candidate_preprocess_smoke_report.json",
    )
    parser.add_argument(
        "--preprocess-summary",
        default="review/v3_2_recall_fix/v3_2_train_candidate_preprocess_smoke_summary.md",
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


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
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


def feature_stats_by(rows: list[tuple[dict[str, str], np.ndarray]], group_field: str) -> dict[str, Any]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, feature in rows:
        grouped[row.get(group_field, "")].append(feature)
    return {key: feature_stats(values) for key, values in sorted(grouped.items())}


def run_dataset_check(
    rows: list[dict[str, str]],
    columns: list[str],
    merge_report: dict[str, Any],
    expected_rows: int,
    expected_labels: dict[str, int],
    label_hash_before: dict[str, str],
    label_hash_after: dict[str, str],
    candidate_hash_before: str,
    candidate_hash_after: str,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_root = ROOT / "data" / "raw"
    missing_fields = [field for field in REQUIRED_FIELDS if field not in columns]
    if missing_fields:
        errors.append({"kind": "missing_required_fields", "fields": missing_fields})

    labels = [row.get("label", "") for row in rows]
    sample_ids = [row.get("sample_id", "") for row in rows]
    review_ids = [row.get("review_id", "") for row in rows]
    audio_cache: dict[str, dict[str, Any]] = {}
    bad_rows: list[dict[str, Any]] = []
    window_duration_values: list[float] = []

    for row in rows:
        review_id = row.get("review_id", "")
        audio_path = resolve_path(row.get("source_audio_path", ""))
        json_path = resolve_path(row.get("source_json_path", ""))
        if is_under(audio_path, raw_root) or is_under(json_path, raw_root):
            bad_rows.append({"review_id": review_id, "kind": "ml_data_raw_path"})
        if not audio_path.exists():
            bad_rows.append({"review_id": review_id, "kind": "missing_audio", "path": str(audio_path)})
            continue
        if not json_path.exists():
            bad_rows.append({"review_id": review_id, "kind": "missing_json", "path": str(json_path)})
        cache_key = str(audio_path)
        if cache_key not in audio_cache:
            try:
                audio_cache[cache_key] = wav_info(audio_path)
            except Exception as exc:  # pragma: no cover - defensive report
                audio_cache[cache_key] = {"error": str(exc)}
        info = audio_cache[cache_key]
        if "error" in info:
            bad_rows.append({"review_id": review_id, "kind": "wav_header_error", "error": info["error"]})
            continue
        if int(info["sample_rate"]) != 16000:
            bad_rows.append({"review_id": review_id, "kind": "bad_sample_rate", "value": info["sample_rate"]})
        if int(info["channels"]) != 1:
            bad_rows.append({"review_id": review_id, "kind": "bad_channels", "value": info["channels"]})
        start = parse_float(row.get("proposed_window_start_sec", ""))
        end = parse_float(row.get("proposed_window_end_sec", ""))
        duration = end - start if not math.isnan(start) and not math.isnan(end) else math.nan
        if not math.isnan(duration):
            window_duration_values.append(duration)
        if math.isnan(start) or math.isnan(end) or start < 0 or end <= start:
            bad_rows.append({"review_id": review_id, "kind": "bad_proposed_window_time"})
        elif abs((end - start) - 1.0) > 1e-3:
            bad_rows.append({"review_id": review_id, "kind": "window_duration_not_1s", "duration": end - start})
        elif end > float(info["duration_s"]) + 1e-6:
            bad_rows.append({"review_id": review_id, "kind": "window_exceeds_wav_duration", "duration_s": info["duration_s"]})

    if set(labels) - {"cough", "non_cough"}:
        errors.append({"kind": "invalid_labels", "values": sorted(set(labels) - {"cough", "non_cough"})})
    if len(sample_ids) != len(set(sample_ids)):
        errors.append({"kind": "duplicate_sample_id", "count": len(sample_ids) - len(set(sample_ids))})
    if len(review_ids) != len(set(review_ids)):
        errors.append({"kind": "duplicate_review_id", "count": len(review_ids) - len(set(review_ids))})
    if bad_rows:
        errors.append({"kind": "path_or_boundary_failures", "count": len(bad_rows), "examples": bad_rows[:30]})

    forbidden_counts = {
        "p03": sum(1 for row in rows if row.get("person_id_for_analysis") == "p03"),
        "p02": sum(1 for row in rows if row.get("person_id_for_analysis") == "p02"),
        "long_quiet_eval": sum(1 for row in rows if row.get("split_role_suggested") == "long_quiet_eval"),
        "distance_extrap_eval": sum(1 for row in rows if row.get("split_role_suggested") == "distance_extrap_eval"),
    }
    if any(forbidden_counts.values()):
        errors.append({"kind": "forbidden_holdout_rows_present", "counts": forbidden_counts})

    train_paths = [resolve_path(path) for path in FORBIDDEN_TRAIN_LABEL_PATHS]
    train_outputs_found = [str(path) for path in train_paths if path.exists()]
    if train_outputs_found:
        errors.append({"kind": "forbidden_train_label_exists", "paths": train_outputs_found})

    sample_rate_counts = counts([str(info.get("sample_rate")) for info in audio_cache.values() if "sample_rate" in info])
    channels_counts = counts([str(info.get("channels")) for info in audio_cache.values() if "channels" in info])
    distribution = {
        "label_distribution": counts(labels),
        "candidate_role_distribution": counts([row.get("candidate_role", "") for row in rows]),
        "person_distribution": counts([row.get("person_id_for_analysis", "") for row in rows]),
        "source_actor_type_distribution": counts([row.get("source_actor_type", "") for row in rows]),
        "scenario_distribution": counts([row.get("scenario_canonical", "") for row in rows]),
        "split_role_distribution": counts([row.get("split_role_suggested", "") for row in rows]),
        "candidate_role_by_label": crosstab(rows, "candidate_role", "label"),
    }
    validation = {
        "row_count_matches_merge_report": len(rows) == expected_rows,
        "labels_only_cough_non_cough": set(labels).issubset({"cough", "non_cough"}),
        "label_distribution_matches_merge_report": counts(labels) == expected_labels,
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)),
        "review_id_unique": len(review_ids) == len(set(review_ids)),
        "v32rev_0131_not_present": "v32rev_0131" not in set(review_ids),
        "all_paths_and_bounds_ok": not bad_rows,
        "sample_rate_all_16000": set(sample_rate_counts.keys()).issubset({"16000"}) and bool(sample_rate_counts),
        "channels_all_1": set(channels_counts.keys()).issubset({"1"}) and bool(channels_counts),
        "no_ml_data_raw_paths": not any(item.get("kind") == "ml_data_raw_path" for item in bad_rows),
        "no_p03_p02_long_quiet_distance": not any(forbidden_counts.values()),
        "old_label_hashes_unchanged": label_hash_before == label_hash_after,
        "candidate_csv_unchanged": candidate_hash_before == candidate_hash_after,
        "no_formal_audio_labels_v3_2_train": not train_outputs_found,
        "merge_report_validation_pass": merge_report.get("validation_result") == "pass",
        "training_not_run": True,
        "ml_data_raw_not_written": True,
    }
    return {
        "stage": "v3.2-CHECK-1-DATASET",
        "checked_rows": len(rows),
        "expected_rows": expected_rows,
        "expected_label_distribution": expected_labels,
        "unique_audio_paths": len(audio_cache),
        "required_fields": REQUIRED_FIELDS,
        "missing_fields": missing_fields,
        "distribution": distribution,
        "sample_rate_counts": sample_rate_counts,
        "channels_counts": channels_counts,
        "window_duration_summary": {
            "count": len(window_duration_values),
            "min": min(window_duration_values) if window_duration_values else None,
            "max": max(window_duration_values) if window_duration_values else None,
        },
        "forbidden_holdout_counts": forbidden_counts,
        "train_outputs_found": train_outputs_found,
        "old_label_hash_before": label_hash_before,
        "old_label_hash_after": label_hash_after,
        "candidate_hash_before": candidate_hash_before,
        "candidate_hash_after": candidate_hash_after,
        "errors": errors,
        "warnings": warnings,
        "validation": validation,
        "result": "pass" if all(validation.values()) and not errors else "fail",
    }


def run_preprocess_smoke(
    rows: list[dict[str, str]],
    audio_config: dict[str, Any],
    frontend_check: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    feature_rows: list[tuple[dict[str, str], np.ndarray]] = []
    bad_shape_count = 0
    nan_inf_count = 0
    failed_rows = 0
    shape_counts: Counter[str] = Counter()

    for row in rows:
        review_id = row.get("review_id", "")
        try:
            start = parse_float(row["proposed_window_start_sec"])
            end = parse_float(row["proposed_window_end_sec"])
            segment = load_audio_segment(
                resolve_path(row["source_audio_path"]),
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
            feature = log_mel_feature(windows[0][2], audio_config)
        except Exception as exc:
            failed_rows += 1
            errors.append({"review_id": review_id, "kind": "feature_extract_failed", "error": str(exc)})
            continue

        shape = tuple(int(value) for value in feature.shape)
        shape_counts[str(list(shape))] += 1
        if shape != EXPECTED_FEATURE_SHAPE:
            bad_shape_count += 1
            errors.append({"review_id": review_id, "kind": "bad_feature_shape", "shape": list(shape)})
        if not np.isfinite(feature).all():
            nan_inf_count += 1
            errors.append({"review_id": review_id, "kind": "nan_inf_feature"})
        feature_rows.append((row, feature))

    validation = {
        "frontend_config_pass": frontend_check["result"] == "pass",
        "checked_rows_matches_expected": len(rows) == expected_rows,
        "feature_success_rows_matches_expected": len(feature_rows) == expected_rows,
        "failed_rows_is_0": failed_rows == 0,
        "feature_shape_all_40_101": dict(shape_counts) == {str([40, 101]): expected_rows},
        "nan_inf_feature_count_is_0": nan_inf_count == 0,
        "bad_shape_feature_count_is_0": bad_shape_count == 0,
        "processed_feature_files_not_saved": True,
    }
    return {
        "stage": "v3.2-CHECK-1-PREPROCESS-SMOKE",
        "audio_frontend": "board_htk_no_norm_v1",
        "audio_config": audio_config,
        "frontend_check": frontend_check,
        "checked_rows": len(rows),
        "expected_rows": expected_rows,
        "feature_success_rows": len(feature_rows),
        "failed_rows": failed_rows,
        "expected_feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "feature_shape_counts": dict(shape_counts),
        "nan_inf_feature_count": nan_inf_count,
        "bad_shape_feature_count": bad_shape_count,
        "feature_stats": feature_stats([feature for _, feature in feature_rows]),
        "feature_stats_by_label": feature_stats_by(feature_rows, "label"),
        "feature_stats_by_candidate_role": feature_stats_by(feature_rows, "candidate_role"),
        "errors": errors,
        "validation": validation,
        "result": "pass" if all(validation.values()) and not errors else "fail",
    }


def write_dataset_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Train-Candidate Dataset Check Summary",
        "",
        f"- result: `{report['result']}`",
        f"- checked_rows: `{report['checked_rows']}`",
        f"- unique_audio_paths: `{report['unique_audio_paths']}`",
        f"- errors: `{len(report['errors'])}`",
        "",
        "## Label Distribution",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for key, value in report["distribution"]["label_distribution"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Validation", "", "| check | result |", "| --- | --- |"])
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preprocess_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Train-Candidate Preprocess Smoke Summary",
        "",
        f"- result: `{report['result']}`",
        f"- audio_frontend: `{report['audio_frontend']}`",
        f"- checked_rows: `{report['checked_rows']}`",
        f"- feature_success_rows: `{report['feature_success_rows']}`",
        f"- failed_rows: `{report['failed_rows']}`",
        f"- expected_feature_shape: `{report['expected_feature_shape']}`",
        f"- feature_shape_counts: `{report['feature_shape_counts']}`",
        "",
        "## Feature Stats",
        "",
        f"`{report['feature_stats']}`",
        "",
        "## Validation",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
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

    old_label_hash_before = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    candidate_hash_before = sha256_or_missing(labels_path)
    rows, columns = read_csv(labels_path)
    merge_report = read_json(merge_report_path)
    expected_rows = int(merge_report.get("row_counts", {}).get("output_rows", len(rows)))
    expected_labels = {
        str(key): int(value)
        for key, value in merge_report.get("label_distribution", counts([row.get("label", "") for row in rows])).items()
    }
    config = read_yaml(config_path)
    audio_config = dict(config["audio"])
    frontend_check = validate_audio_config(audio_config)

    dataset_report = run_dataset_check(
        rows,
        columns,
        merge_report,
        expected_rows,
        expected_labels,
        old_label_hash_before,
        {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS},
        candidate_hash_before,
        sha256_or_missing(labels_path),
    )
    preprocess_report = run_preprocess_smoke(rows, audio_config, frontend_check, expected_rows)

    # Final no-change checks after both phases.
    old_label_hash_after = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    candidate_hash_after = sha256_or_missing(labels_path)
    dataset_report["old_label_hash_after"] = old_label_hash_after
    dataset_report["candidate_hash_after"] = candidate_hash_after
    dataset_report["validation"]["old_label_hashes_unchanged"] = old_label_hash_before == old_label_hash_after
    dataset_report["validation"]["candidate_csv_unchanged"] = candidate_hash_before == candidate_hash_after
    dataset_report["result"] = "pass" if all(dataset_report["validation"].values()) and not dataset_report["errors"] else "fail"
    preprocess_report["validation"]["old_label_hashes_unchanged"] = old_label_hash_before == old_label_hash_after
    preprocess_report["validation"]["candidate_csv_unchanged"] = candidate_hash_before == candidate_hash_after
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
                "feature_success_rows": preprocess_report["feature_success_rows"],
                "failed_rows": preprocess_report["failed_rows"],
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
