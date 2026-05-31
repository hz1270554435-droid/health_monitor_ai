from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_COLUMNS = [
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
REQUIRED_CANDIDATE_COLUMNS = [
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
    "fpfix_cough_prob",
    "notes",
]
REQUIRED_DRY_RUN_COLUMNS = [
    "sample_id",
    "review_id",
    "proposed_window_start_sec",
    "proposed_window_end_sec",
    "event_start_sec",
    "event_end_sec",
    "proposed_merge_role",
    "merge_include",
    "sampling_bucket",
    "sampling_weight_hint",
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
        description="Build a v3.2 train-candidate label CSV from an approved merge dry-run plan."
    )
    parser.add_argument("--candidate", default="review/v3_2_recall_fix/audio_labels_v3_2_candidate_recall_fix.csv")
    parser.add_argument("--dry-run", default="review/v3_2_recall_fix/v3_2_candidate_merge_dry_run.csv")
    parser.add_argument("--dry-run-report", default="review/v3_2_recall_fix/v3_2_candidate_merge_dry_run_report.json")
    parser.add_argument("--policy-md", default="review/v3_2_recall_fix/v3_2_candidate_to_train_merge_policy.md")
    parser.add_argument(
        "--out",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run.csv",
    )
    parser.add_argument(
        "--report",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run_report.json",
    )
    parser.add_argument(
        "--summary-md",
        default="review/v3_2_recall_fix/audio_labels_v3_2_train_candidate_from_dry_run_summary.md",
    )
    parser.add_argument("--merge-policy-version", default="v3_2_merge_policy_dry_run_v1_20260528")
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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


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


def validate_columns(columns: list[str], required: list[str], source_name: str) -> list[str]:
    return [f"{source_name} missing required column: {column}" for column in required if column not in columns]


def crosstab(rows: list[dict[str, str]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = {}
    for row in rows:
        key = row.get(left, "")
        if key not in table:
            table[key] = Counter()
        table[key][row.get(right, "")] += 1
    return {key: dict(value.most_common()) for key, value in sorted(table.items())}


def build_output_row(candidate: dict[str, str], dry: dict[str, str], merge_policy_version: str) -> dict[str, str]:
    notes = [
        "train_candidate_from_dry_run_only",
        f"dry_run_sampling_bucket={dry.get('sampling_bucket', '')}",
        f"sampling_weight_hint={dry.get('sampling_weight_hint', '')}",
        f"candidate_notes={candidate.get('notes', '')}",
        f"dry_run_notes={dry.get('notes', '')}",
    ]
    return {
        "sample_id": candidate["sample_id"],
        "review_id": candidate["review_id"],
        "source_audio_path": candidate["source_audio_path"],
        "source_json_path": candidate["source_json_path"],
        "label": candidate["label"],
        "candidate_role": candidate["candidate_role"],
        "downstream_decision": candidate["downstream_decision"],
        "person_id_for_analysis": candidate["person_id_for_analysis"],
        "source_actor_type": candidate["source_actor_type"],
        "scenario_canonical": candidate["scenario_canonical"],
        "split_role_suggested": candidate["split_role_suggested"],
        "source_domain": candidate["source_domain"],
        "proposed_window_start_sec": dry["proposed_window_start_sec"],
        "proposed_window_end_sec": dry["proposed_window_end_sec"],
        "event_start_sec": dry["event_start_sec"],
        "event_end_sec": dry["event_end_sec"],
        "fpfix_cough_prob": candidate["fpfix_cough_prob"],
        "merge_policy_version": merge_policy_version,
        "notes": ";".join(notes),
    }


def write_summary(path: Path, report: dict[str, Any]) -> None:
    def table(title: str, values: dict[str, int]) -> list[str]:
        lines = [f"## {title}", "", "| value | count |", "| --- | ---: |"]
        for key, value in values.items():
            lines.append(f"| {key} | {value} |")
        return lines

    lines = [
        "# v3.2 Train Candidate From Dry-Run Summary",
        "",
        "This file is a candidate training-label addon produced from the approved dry-run rows only. It is not the final `data/labels/audio_labels_v3_2_train.csv`, does not train, does not merge into old labels, and does not modify raw WAV/JSON or `ml/data/raw/`.",
        "",
        "## Outputs",
        "",
        f"- train_candidate_csv: `{report['outputs']['train_candidate_csv']}`",
        f"- report_json: `{report['outputs']['report_json']}`",
        f"- summary_md: `{report['outputs']['summary_md']}`",
        "",
        "## Counts",
        "",
        f"- output_rows: `{report['row_counts']['output_rows']}`",
        f"- selected_dry_run_rows: `{report['row_counts']['selected_dry_run_rows']}`",
        "",
    ]
    lines.extend(table("Label Distribution", report["label_distribution"]))
    lines.extend([""])
    lines.extend(table("Candidate Role Distribution", report["candidate_role_distribution"]))
    lines.extend([""])
    lines.extend(table("Person Distribution", report["person_distribution"]))
    lines.extend([""])
    lines.extend(table("Scenario Distribution", report["scenario_distribution"]))
    lines.extend(["", "## Holdout Confirmation", ""])
    for key, value in report["holdout_confirmation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Validation", "", "| check | result |", "| --- | --- |"])
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            f"Validation result: `{report['validation_result']}`",
            "",
            "## Next Step",
            "",
            report["recommended_next_step"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_path = resolve_path(args.candidate)
    dry_run_path = resolve_path(args.dry_run)
    dry_report_path = resolve_path(args.dry_run_report)
    policy_path = resolve_path(args.policy_md)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    for required in (candidate_path, dry_run_path, dry_report_path, policy_path):
        if not required.exists():
            raise FileNotFoundError(required)
    for output in (out_path, report_path, summary_path):
        ensure_new_file(output, args.overwrite)

    old_hash_before = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    candidate_hash_before = sha256_or_missing(candidate_path)
    dry_hash_before = sha256_or_missing(dry_run_path)

    candidate_rows, candidate_columns = read_csv(candidate_path)
    dry_rows, dry_columns = read_csv(dry_run_path)
    dry_report = read_json(dry_report_path)

    errors: list[str] = []
    errors.extend(validate_columns(candidate_columns, REQUIRED_CANDIDATE_COLUMNS, "candidate"))
    errors.extend(validate_columns(dry_columns, REQUIRED_DRY_RUN_COLUMNS, "dry_run"))
    if dry_report.get("validation_result") != "pass":
        errors.append("dry-run report validation_result is not pass")

    candidate_by_review = {row["review_id"]: row for row in candidate_rows}
    selected_dry = [
        row
        for row in dry_rows
        if parse_bool(row.get("merge_include", "")) and row.get("proposed_merge_role") == "train_addon"
    ]
    output_rows: list[dict[str, str]] = []
    for dry in selected_dry:
        review_id = dry["review_id"]
        candidate = candidate_by_review.get(review_id)
        if candidate is None:
            errors.append(f"dry-run selected review_id missing in candidate CSV: {review_id}")
            continue
        output_rows.append(build_output_row(candidate, dry, args.merge_policy_version))

    audio_cache: dict[str, dict[str, float | int]] = {}
    path_failures: list[dict[str, Any]] = []
    bad_bounds = 0
    bad_audio = 0
    bad_header = 0
    missing_json = 0
    for row in output_rows:
        review_id = row["review_id"]
        audio_path = resolve_path(row["source_audio_path"])
        json_path = resolve_path(row["source_json_path"]) if row["source_json_path"].strip() else None
        if not audio_path.exists():
            bad_audio += 1
            path_failures.append({"review_id": review_id, "kind": "missing_audio", "path": str(audio_path)})
            continue
        if json_path is not None and not json_path.exists():
            missing_json += 1
            path_failures.append({"review_id": review_id, "kind": "missing_json", "path": str(json_path)})
        key = str(audio_path)
        if key not in audio_cache:
            try:
                audio_cache[key] = wav_info(audio_path)
            except Exception as exc:  # pragma: no cover - audit defensive path
                audio_cache[key] = {"error": str(exc)}
        info = audio_cache[key]
        if "error" in info:
            bad_audio += 1
            path_failures.append({"review_id": review_id, "kind": "wav_header_error", "error": info["error"]})
            continue
        if int(info["sample_rate"]) != 16000 or int(info["channels"]) != 1:
            bad_header += 1
            path_failures.append(
                {
                    "review_id": review_id,
                    "kind": "bad_wav_header",
                    "sample_rate": info["sample_rate"],
                    "channels": info["channels"],
                }
            )
        start = parse_float(row["proposed_window_start_sec"])
        end = parse_float(row["proposed_window_end_sec"])
        if math.isnan(start) or math.isnan(end) or start < 0 or end <= start or end > float(info["duration_s"]) + 1e-6:
            bad_bounds += 1
            path_failures.append(
                {
                    "review_id": review_id,
                    "kind": "bad_proposed_window_bounds",
                    "start": row["proposed_window_start_sec"],
                    "end": row["proposed_window_end_sec"],
                    "duration_s": info["duration_s"],
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    old_hash_after = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    candidate_hash_after = sha256_or_missing(candidate_path)
    dry_hash_after = sha256_or_missing(dry_run_path)
    forbidden_train_outputs = [str(resolve_path(path)) for path in FORBIDDEN_TRAIN_LABEL_PATHS if resolve_path(path).exists()]

    label_counts = counts([row["label"] for row in output_rows])
    role_counts = counts([row["candidate_role"] for row in output_rows])
    person_counts = counts([row["person_id_for_analysis"] for row in output_rows])
    scenario_counts = counts([row["scenario_canonical"] for row in output_rows])
    split_counts = counts([row["split_role_suggested"] for row in output_rows])
    source_actor_counts = counts([row["source_actor_type"] for row in output_rows])
    sample_ids = [row["sample_id"] for row in output_rows]
    review_ids = [row["review_id"] for row in output_rows]
    label_values = set(row["label"] for row in output_rows)
    expected_selected_rows = int(dry_report.get("merge_counts", {}).get("merge_include_true", len(selected_dry)))
    expected_label_counts = counts(
        [
            candidate_by_review[dry["review_id"]]["label"]
            for dry in selected_dry
            if dry.get("review_id") in candidate_by_review
        ]
    )
    short_window_excluded_ids = set(dry_report.get("short_window_exclusion", {}).get("review_ids", []))

    holdout_confirmation = {
        "p03_in_output": sum(1 for row in output_rows if row["person_id_for_analysis"] == "p03"),
        "p02_in_output": sum(1 for row in output_rows if row["person_id_for_analysis"] == "p02"),
        "long_quiet_eval_in_output": sum(1 for row in output_rows if row["split_role_suggested"] == "long_quiet_eval"),
        "distance_extrap_eval_in_output": sum(1 for row in output_rows if row["split_role_suggested"] == "distance_extrap_eval"),
    }
    validation = {
        "output_rows_match_selected_dry_run_rows": len(output_rows) == len(selected_dry),
        "selected_dry_run_rows_match_dry_run_report": len(selected_dry) == expected_selected_rows,
        "labels_only_cough_non_cough": label_values.issubset({"cough", "non_cough"}),
        "label_distribution_matches_selected_dry_run": label_counts == expected_label_counts,
        "v32rev_0131_not_in_output": "v32rev_0131" not in review_ids,
        "short_window_excluded_ids_not_in_output": not (short_window_excluded_ids & set(review_ids)),
        "p03_not_in_output": holdout_confirmation["p03_in_output"] == 0,
        "p02_not_in_output": holdout_confirmation["p02_in_output"] == 0,
        "long_quiet_eval_not_in_output": holdout_confirmation["long_quiet_eval_in_output"] == 0,
        "distance_extrap_eval_not_in_output": holdout_confirmation["distance_extrap_eval_in_output"] == 0,
        "source_audio_exists": bad_audio == 0,
        "source_json_exists": missing_json == 0,
        "proposed_window_bounds_ok": bad_bounds == 0,
        "sample_rate_16000_channels_1": bad_header == 0,
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)),
        "review_id_unique": len(review_ids) == len(set(review_ids)),
        "old_label_hashes_unchanged": old_hash_before == old_hash_after,
        "candidate_csv_unchanged": candidate_hash_before == candidate_hash_after,
        "dry_run_csv_unchanged": dry_hash_before == dry_hash_after,
        "formal_audio_labels_v3_2_train_not_generated": not forbidden_train_outputs,
        "training_not_run": True,
        "ml_data_raw_not_written": True,
        "raw_wav_json_not_modified": True,
        "audio_not_copied": True,
        "no_script_errors": not errors,
    }
    validation_result = "pass" if all(validation.values()) else "fail"
    report: dict[str, Any] = {
        "stage": "v3.2-MERGE-1",
        "inputs": {
            "candidate_csv": str(candidate_path),
            "dry_run_csv": str(dry_run_path),
            "dry_run_report": str(dry_report_path),
            "policy_md": str(policy_path),
        },
        "outputs": {
            "train_candidate_csv": str(out_path),
            "report_json": str(report_path),
            "summary_md": str(summary_path),
        },
        "merge_policy_version": args.merge_policy_version,
        "row_counts": {
            "candidate_input_rows": len(candidate_rows),
            "dry_run_rows": len(dry_rows),
            "selected_dry_run_rows": len(selected_dry),
            "expected_selected_rows_from_dry_run_report": expected_selected_rows,
            "output_rows": len(output_rows),
        },
        "expected_label_distribution_from_selected_dry_run": expected_label_counts,
        "short_window_excluded_review_ids": sorted(short_window_excluded_ids),
        "label_distribution": label_counts,
        "candidate_role_distribution": role_counts,
        "person_distribution": person_counts,
        "source_actor_type_distribution": source_actor_counts,
        "scenario_distribution": scenario_counts,
        "split_role_distribution": split_counts,
        "holdout_confirmation": holdout_confirmation,
        "path_validation": {
            "unique_source_audio_paths": len(audio_cache),
            "missing_audio_rows": bad_audio,
            "missing_json_rows": missing_json,
            "bad_proposed_window_rows": bad_bounds,
            "bad_wav_header_rows": bad_header,
            "path_failures": path_failures[:50],
            "path_failures_truncated": len(path_failures) > 50,
        },
        "old_label_hash_before": old_hash_before,
        "old_label_hash_after": old_hash_after,
        "candidate_hash_before": candidate_hash_before,
        "candidate_hash_after": candidate_hash_after,
        "dry_run_hash_before": dry_hash_before,
        "dry_run_hash_after": dry_hash_after,
        "forbidden_train_outputs_found": forbidden_train_outputs,
        "errors": errors,
        "validation": validation,
        "validation_result": validation_result,
        "not_executed": [
            "training",
            "final data/labels/audio_labels_v3_2_train.csv generation",
            "old label modification",
            "candidate CSV modification",
            "dry-run CSV modification",
            "raw WAV/JSON modification",
            "audio copy",
            "ml/data/raw write",
        ],
        "recommended_next_step": (
            "Run dataset check and board_htk_no_norm_v1 preprocess smoke on this train-candidate CSV. "
            "Do not train until those checks pass and the final train-label merge target is explicitly approved."
        ),
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
