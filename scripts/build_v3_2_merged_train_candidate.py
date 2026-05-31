from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OLD_LABEL_PATHS = [
    "data/labels/audio_labels_v3.csv",
    "data/labels/audio_labels_v3_1_fpfix_train.csv",
]
FORBIDDEN_TRAIN_LABEL_PATHS = [
    "data/labels/audio_labels_v3_2_train.csv",
    "review/v3_2_recall_fix/audio_labels_v3_2_train.csv",
]
ADDON_STAGE = "v3_2_recall_fix_addon"
BASE_STAGE = "v3_1_fpfix_base"
ADDON_REQUIRED_COLUMNS = [
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
EXTRA_COLUMNS = [
    "sample_id",
    "source_stage",
    "source_audio_path",
    "source_json_path",
    "candidate_role",
    "downstream_decision",
    "person_id_for_analysis",
    "source_actor_type",
    "scenario_canonical",
    "split_role_suggested",
    "proposed_window_start_sec",
    "proposed_window_end_sec",
    "event_start_sec",
    "event_end_sec",
    "fpfix_cough_prob",
    "merge_policy_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a non-final v3.2 merged train-candidate label CSV from v3.1-fpfix base plus checked v3.2 addon."
    )
    parser.add_argument("--base", default="data/labels/audio_labels_v3_1_fpfix_train.csv")
    parser.add_argument("--addon", default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run.csv")
    parser.add_argument(
        "--dataset-report",
        default="review/v3_2_recall_fix/v3_2_train_candidate_dataset_check_report.json",
    )
    parser.add_argument(
        "--preprocess-report",
        default="review/v3_2_recall_fix/v3_2_train_candidate_preprocess_smoke_report.json",
    )
    parser.add_argument(
        "--out",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_merged_candidate.csv",
    )
    parser.add_argument(
        "--report",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_merged_candidate_report.json",
    )
    parser.add_argument(
        "--summary-md",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_merged_candidate_summary.md",
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


def parse_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else math.nan
    except ValueError:
        return math.nan


def format_sec(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


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


def source_audio_for_base(row: dict[str, str]) -> str:
    return first_nonempty(row, ["canonical_audio_path", "source_audio", "original_path", "audio_file"])


def start_end_for_row(row: dict[str, str]) -> tuple[float, float]:
    start = parse_float(first_nonempty(row, ["source_start_time", "window_start_sec", "start_time", "proposed_window_start_sec"]))
    end = parse_float(first_nonempty(row, ["source_end_time", "window_end_sec", "end_time", "proposed_window_end_sec"]))
    return start, end


def wav_info(path: Path) -> dict[str, float | int]:
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


def make_columns(base_columns: list[str], addon_columns: list[str]) -> list[str]:
    columns: list[str] = []
    for column in base_columns + EXTRA_COLUMNS + addon_columns:
        if column not in columns:
            columns.append(column)
    return columns


def normalize_base_row(row: dict[str, str], columns: list[str]) -> dict[str, str]:
    out = {column: row.get(column, "") for column in columns}
    out["sample_id"] = out.get("sample_id") or row.get("clip_id", "")
    out["source_stage"] = out.get("source_stage") or BASE_STAGE
    out["source_audio_path"] = out.get("source_audio_path") or source_audio_for_base(row)
    out["source_json_path"] = out.get("source_json_path", "")
    return out


def normalize_addon_row(row: dict[str, str], columns: list[str], final_sample_id: str) -> dict[str, str]:
    start = parse_float(row["proposed_window_start_sec"])
    end = parse_float(row["proposed_window_end_sec"])
    hard_negative = row.get("candidate_role") == "hard_negative_candidate" or row.get("downstream_decision") == "hard_negative"
    session_id = Path(row["source_audio_path"]).stem.replace("_audio", "")
    notes = [
        row.get("notes", ""),
        "merged_candidate_only",
        "not_final_train_label",
    ]
    out = {column: "" for column in columns}
    out.update(
        {
            "sample_id": final_sample_id,
            "clip_id": final_sample_id,
            "session_id": session_id,
            "person_id": row.get("person_id_for_analysis", ""),
            "start_time": format_sec(start),
            "end_time": format_sec(end),
            "label": row.get("label", ""),
            "audio_file": row.get("source_audio_path", ""),
            "split": "train",
            "hard_negative": str(bool(hard_negative)),
            "sample_weight": "1.0",
            "source": "v3_2_recall_fix_manual_review",
            "source_domain": row.get("source_domain", ""),
            "change_reason": "v3_2_recall_fix_train_candidate_addon",
            "hard_negative_source": "v3_2_recall_fix" if hard_negative else "",
            "negative_type": row.get("downstream_decision", "") if row.get("label") == "non_cough" else "",
            "mining_role": row.get("candidate_role", ""),
            "row_origin": ADDON_STAGE,
            "original_path": row.get("source_audio_path", ""),
            "source_audio": row.get("source_audio_path", ""),
            "canonical_audio_path": row.get("source_audio_path", ""),
            "source_start_time": format_sec(start),
            "source_end_time": format_sec(end),
            "window_start_sec": format_sec(start),
            "window_end_sec": format_sec(end),
            "manual_decision": row.get("label", ""),
            "manual_subtype_norm": row.get("downstream_decision", ""),
            "candidate_id": final_sample_id,
            "review_clip_is_training_source": "False",
            "canonical_training_window": "v3_2_proposed_1s_window",
            "notes": ";".join(part for part in notes if part),
            "review_batch": "v3_2_recall_fix",
            "review_id": row.get("review_id", ""),
            "review_file": "review_plan_combined_corrected_qa_recheck_applied.csv",
            "derived_review_class": row.get("downstream_decision", ""),
            "source_stage": ADDON_STAGE,
            "source_audio_path": row.get("source_audio_path", ""),
            "source_json_path": row.get("source_json_path", ""),
            "candidate_role": row.get("candidate_role", ""),
            "downstream_decision": row.get("downstream_decision", ""),
            "person_id_for_analysis": row.get("person_id_for_analysis", ""),
            "source_actor_type": row.get("source_actor_type", ""),
            "scenario_canonical": row.get("scenario_canonical", ""),
            "split_role_suggested": row.get("split_role_suggested", ""),
            "proposed_window_start_sec": format_sec(start),
            "proposed_window_end_sec": format_sec(end),
            "event_start_sec": row.get("event_start_sec", ""),
            "event_end_sec": row.get("event_end_sec", ""),
            "fpfix_cough_prob": row.get("fpfix_cough_prob", ""),
            "merge_policy_version": row.get("merge_policy_version", ""),
        }
    )
    return out


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    raw_root = ROOT / "data" / "raw"
    audio_cache: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    stage_channels: dict[str, Counter[str]] = defaultdict(Counter)
    stage_sample_rates: dict[str, Counter[str]] = defaultdict(Counter)
    ml_data_raw_paths = 0

    for row in rows:
        sample_id = row.get("sample_id", "")
        stage = row.get("source_stage", "")
        audio_value = row.get("source_audio_path", "")
        audio_path = resolve_path(audio_value)
        if is_under(audio_path, raw_root):
            ml_data_raw_paths += 1
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
        sample_rates[str(info["sample_rate"])] += 1
        channels[str(info["channels"])] += 1
        stage_sample_rates[stage][str(info["sample_rate"])] += 1
        stage_channels[stage][str(info["channels"])] += 1
        start, end = start_end_for_row(row)
        if math.isnan(start) or math.isnan(end) or start < 0 or end <= start:
            failures.append({"sample_id": sample_id, "kind": "bad_window_time", "start": start, "end": end})
        elif end > float(info["duration_s"]) + 1e-6:
            failures.append(
                {
                    "sample_id": sample_id,
                    "kind": "window_exceeds_wav_duration",
                    "start": start,
                    "end": end,
                    "duration_s": info["duration_s"],
                }
            )

    return {
        "unique_audio_paths": len(audio_cache),
        "sample_rate_counts": dict(sample_rates.most_common()),
        "channels_counts": dict(channels.most_common()),
        "sample_rate_counts_by_stage": {key: dict(value.most_common()) for key, value in sorted(stage_sample_rates.items())},
        "channels_counts_by_stage": {key: dict(value.most_common()) for key, value in sorted(stage_channels.items())},
        "ml_data_raw_path_rows": ml_data_raw_paths,
        "failures": failures[:50],
        "failure_count": len(failures),
        "failures_truncated": len(failures) > 50,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / float(denominator)


def write_summary(path: Path, report: dict[str, Any]) -> None:
    def table(title: str, values: dict[str, int]) -> list[str]:
        lines = [f"## {title}", "", "| value | count |", "| --- | ---: |"]
        for key, value in values.items():
            lines.append(f"| {key} | {value} |")
        return lines

    lines = [
        "# v3.2 Merged Train Candidate Summary",
        "",
        "This is a non-final merged train-candidate label file. It does not create `data/labels/audio_labels_v3_2_train.csv`, train, merge into old labels, modify raw WAV/JSON, copy audio, or write `ml/data/raw/`.",
        "",
        f"- validation_result: `{report['validation_result']}`",
        f"- ready_for_final_label_publication: `{report['ready_for_final_label_publication']}`",
        f"- base_rows: `{report['row_counts']['base_rows']}`",
        f"- addon_rows: `{report['row_counts']['addon_rows']}`",
        f"- merged_rows: `{report['row_counts']['merged_rows']}`",
        "",
    ]
    lines.extend(table("Merged Label Distribution", report["label_distribution"]["merged"]))
    lines.extend([""])
    lines.extend(table("Source Stage Distribution", report["source_stage_distribution"]))
    lines.extend([""])
    lines.extend(table("Addon Candidate Role Distribution", report["addon_distribution"]["candidate_role"]))
    lines.extend(
        [
            "",
            "## Channel Check",
            "",
            f"- sample_rate_counts: `{report['path_boundary_validation']['sample_rate_counts']}`",
            f"- channels_counts: `{report['path_boundary_validation']['channels_counts']}`",
            f"- legacy_base_channel_caveat: `{report['legacy_base_channel_caveat']}`",
            "",
            "## Validation",
            "",
            "| check | result |",
            "| --- | --- |",
        ]
    )
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Next Step", "", report["recommended_next_step"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_path = resolve_path(args.base)
    addon_path = resolve_path(args.addon)
    dataset_report_path = resolve_path(args.dataset_report)
    preprocess_report_path = resolve_path(args.preprocess_report)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    for required in (base_path, addon_path, dataset_report_path, preprocess_report_path):
        if not required.exists():
            raise FileNotFoundError(required)
    for output in (out_path, report_path, summary_path):
        ensure_new_file(output, args.overwrite)

    old_hash_before = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    base_hash_before = sha256_or_missing(base_path)
    addon_hash_before = sha256_or_missing(addon_path)
    dataset_report = read_json(dataset_report_path)
    preprocess_report = read_json(preprocess_report_path)
    base_rows, base_columns = read_csv(base_path)
    addon_rows, addon_columns = read_csv(addon_path)
    columns = make_columns(base_columns, addon_columns)

    missing_addon_columns = [column for column in ADDON_REQUIRED_COLUMNS if column not in addon_columns]
    used_ids: set[str] = set()
    merged_rows: list[dict[str, str]] = []
    for row in base_rows:
        out = normalize_base_row(row, columns)
        if out.get("sample_id"):
            used_ids.add(out["sample_id"])
        merged_rows.append(out)

    sample_id_conflicts = 0
    for row in addon_rows:
        sample_id = row["sample_id"]
        if sample_id in used_ids:
            sample_id = f"v3_2_{sample_id}"
            sample_id_conflicts += 1
        used_ids.add(sample_id)
        merged_rows.append(normalize_addon_row(row, columns, sample_id))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(merged_rows)

    old_hash_after = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    base_hash_after = sha256_or_missing(base_path)
    addon_hash_after = sha256_or_missing(addon_path)
    path_validation = validate_rows(merged_rows)
    forbidden_train_outputs = [str(resolve_path(path)) for path in FORBIDDEN_TRAIN_LABEL_PATHS if resolve_path(path).exists()]

    labels_base = counts([row.get("label", "") for row in base_rows])
    labels_addon = counts([row.get("label", "") for row in addon_rows])
    labels_merged = counts([row.get("label", "") for row in merged_rows])
    source_stage_counts = counts([row.get("source_stage", "") for row in merged_rows])
    addon_output_rows = [row for row in merged_rows if row.get("source_stage") == ADDON_STAGE]
    sample_ids = [row.get("sample_id", "") for row in merged_rows]
    merged_label_values = set(row.get("label", "") for row in merged_rows)
    addon_forbidden_counts = {
        "p03": sum(1 for row in addon_output_rows if row.get("person_id_for_analysis") == "p03"),
        "p02": sum(1 for row in addon_output_rows if row.get("person_id_for_analysis") == "p02"),
        "long_quiet_eval": sum(1 for row in addon_output_rows if row.get("split_role_suggested") == "long_quiet_eval"),
        "distance_extrap_eval": sum(1 for row in addon_output_rows if row.get("split_role_suggested") == "distance_extrap_eval"),
    }
    hard_negative_count = labels_addon and counts([row.get("candidate_role", "") for row in addon_rows]).get("hard_negative_candidate", 0)
    recall_fix_count = labels_addon and counts([row.get("candidate_role", "") for row in addon_rows]).get("positive_recall_fix_candidate", 0)
    strict_channels_all_1 = set(path_validation["channels_counts"].keys()).issubset({"1"}) and bool(
        path_validation["channels_counts"]
    )
    legacy_base_channel_caveat = (
        not strict_channels_all_1
        and path_validation["channels_counts_by_stage"].get(ADDON_STAGE) == {"1": len(addon_rows)}
        and "2" in path_validation["channels_counts_by_stage"].get(BASE_STAGE, {})
    )

    validation = {
        "precondition_dataset_check_pass": dataset_report.get("result") == "pass",
        "precondition_preprocess_smoke_pass": preprocess_report.get("result") == "pass",
        "addon_rows_is_93": len(addon_rows) == 93,
        "merged_rows_equal_base_plus_addon": len(merged_rows) == len(base_rows) + len(addon_rows),
        "labels_only_cough_non_cough": merged_label_values.issubset({"cough", "non_cough"}),
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)),
        "source_audio_paths_exist_and_bounds_ok": path_validation["failure_count"] == 0,
        "sample_rate_all_16000": set(path_validation["sample_rate_counts"].keys()).issubset({"16000"}) and bool(path_validation["sample_rate_counts"]),
        "addon_channels_all_1": path_validation["channels_counts_by_stage"].get(ADDON_STAGE) == {"1": len(addon_rows)},
        "strict_merged_channels_all_1": strict_channels_all_1,
        "p03_p02_long_quiet_distance_not_in_v3_2_addon": not any(addon_forbidden_counts.values()),
        "old_label_hashes_unchanged": old_hash_before == old_hash_after,
        "base_label_file_unchanged": base_hash_before == base_hash_after,
        "addon_label_file_unchanged": addon_hash_before == addon_hash_after,
        "no_final_audio_labels_v3_2_train_generated": not forbidden_train_outputs,
        "ml_data_raw_not_written_or_referenced": path_validation["ml_data_raw_path_rows"] == 0,
        "training_not_run": True,
        "raw_wav_json_not_modified": True,
        "audio_not_copied": True,
        "no_script_errors": not missing_addon_columns,
    }
    critical_validation_keys = [
        key for key in validation if key != "strict_merged_channels_all_1"
    ]
    critical_pass = all(validation[key] for key in critical_validation_keys)
    if critical_pass and validation["strict_merged_channels_all_1"]:
        validation_result = "pass"
    elif critical_pass and legacy_base_channel_caveat:
        validation_result = "pass_with_legacy_base_channel_caveat"
    else:
        validation_result = "fail"

    base_cough = labels_base.get("cough", 0)
    base_non_cough = labels_base.get("non_cough", 0)
    merged_cough = labels_merged.get("cough", 0)
    merged_non_cough = labels_merged.get("non_cough", 0)
    report: dict[str, Any] = {
        "stage": "v3.2-MERGE-2",
        "inputs": {
            "base_labels": str(base_path),
            "addon_labels": str(addon_path),
            "dataset_report": str(dataset_report_path),
            "preprocess_report": str(preprocess_report_path),
        },
        "outputs": {
            "merged_candidate_csv": str(out_path),
            "report_json": str(report_path),
            "summary_md": str(summary_path),
        },
        "row_counts": {
            "base_rows": len(base_rows),
            "addon_rows": len(addon_rows),
            "merged_rows": len(merged_rows),
        },
        "label_distribution": {
            "base": labels_base,
            "addon": labels_addon,
            "merged": labels_merged,
        },
        "class_balance_change_after_addon": {
            "base_cough": base_cough,
            "base_non_cough": base_non_cough,
            "base_non_cough_to_cough_ratio": ratio(base_non_cough, base_cough),
            "addon_cough": labels_addon.get("cough", 0),
            "addon_non_cough": labels_addon.get("non_cough", 0),
            "merged_cough": merged_cough,
            "merged_non_cough": merged_non_cough,
            "merged_non_cough_to_cough_ratio": ratio(merged_non_cough, merged_cough),
            "delta_cough": merged_cough - base_cough,
            "delta_non_cough": merged_non_cough - base_non_cough,
        },
        "source_stage_distribution": source_stage_counts,
        "addon_distribution": {
            "candidate_role": counts([row.get("candidate_role", "") for row in addon_rows]),
            "person_id_for_analysis": counts([row.get("person_id_for_analysis", "") for row in addon_rows]),
            "source_actor_type": counts([row.get("source_actor_type", "") for row in addon_rows]),
            "scenario_canonical": counts([row.get("scenario_canonical", "") for row in addon_rows]),
            "split_role_suggested": counts([row.get("split_role_suggested", "") for row in addon_rows]),
            "candidate_role_by_label": crosstab(addon_rows, "candidate_role", "label"),
        },
        "hard_negative_to_positive_recall_ratio": ratio(hard_negative_count, recall_fix_count),
        "positive_recall_to_hard_negative_ratio": ratio(recall_fix_count, hard_negative_count),
        "sample_id_conflicts_resolved_with_v3_2_prefix": sample_id_conflicts,
        "addon_forbidden_holdout_counts": addon_forbidden_counts,
        "path_boundary_validation": path_validation,
        "legacy_base_channel_caveat": legacy_base_channel_caveat,
        "missing_addon_columns": missing_addon_columns,
        "old_label_hash_before": old_hash_before,
        "old_label_hash_after": old_hash_after,
        "base_hash_before": base_hash_before,
        "base_hash_after": base_hash_after,
        "addon_hash_before": addon_hash_before,
        "addon_hash_after": addon_hash_after,
        "forbidden_train_outputs_found": forbidden_train_outputs,
        "ready_for_final_label_publication": False,
        "validation": validation,
        "validation_result": validation_result,
        "not_executed": [
            "training",
            "one-epoch smoke",
            "final data/labels/audio_labels_v3_2_train.csv generation",
            "old label modification",
            "raw WAV/JSON modification",
            "audio copy",
            "processed feature save",
            "ml/data/raw write",
        ],
        "recommended_next_step": (
            "Review the non-final merged candidate and decide the final publication policy. "
            "Because the preserved v3.1 base contains legacy 16 kHz 2-channel board_live_data_1 rows, "
            "run a merged-candidate preprocess smoke before publishing any final label file."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
