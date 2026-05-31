from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from v3_1_mining_common import stable_id


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COLUMNS = [
    "review_id",
    "audio_file",
    "metadata_json",
    "event_start_sec",
    "event_end_sec",
    "final_manual_decision",
    "final_manual_subtype",
    "final_downstream_decision",
    "final_candidate_role",
    "final_label_included",
    "person_id_for_analysis",
    "source_actor_type",
    "scenario_canonical",
    "split_role_suggested",
    "fpfix_cough_prob",
    "fpfix_non_cough_prob",
    "event_max_prob",
]
OLD_LABEL_PATHS = [
    "data/labels/audio_labels_v3.csv",
    "data/labels/audio_labels_v3_1_fpfix_train.csv",
]
EXPECTED_FINAL_DECISION = {"cough": 154, "non_cough": 165, "exclude": 3}
EXPECTED_FINAL_DOWNSTREAM = {"cough": 154, "hard_negative": 134, "clean_non_cough": 31, "exclude": 3}
EXPECTED_FINAL_ROLE = {
    "hard_negative_candidate": 134,
    "positive_recall_fix_candidate": 120,
    "positive_sanity_candidate": 34,
    "non_cough_candidate": 24,
    "clean_non_cough_smoke_candidate": 7,
    "exclude_until_re_review": 3,
}
NOT_EXECUTED = [
    "training",
    "manifest merge",
    "audio_labels_v3_2_train.csv generation",
    "old label overwrite",
    "raw WAV/JSON modification",
    "audio copy",
    "ml/data/raw write",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v3.2 recall-fix candidate label addon from the QA recheck applied review CSV."
    )
    parser.add_argument(
        "--applied-review",
        default="review/v3_2_recall_fix/review_plan_combined_corrected_qa_recheck_applied.csv",
    )
    parser.add_argument(
        "--out",
        default="review/v3_2_recall_fix/audio_labels_v3_2_candidate_recall_fix.csv",
    )
    parser.add_argument(
        "--report",
        default="review/v3_2_recall_fix/audio_labels_v3_2_candidate_recall_fix_report.json",
    )
    parser.add_argument(
        "--summary-md",
        default="review/v3_2_recall_fix/audio_labels_v3_2_candidate_recall_fix_summary.md",
    )
    parser.add_argument("--source-domain", default="board_live_v3_2_recall_fix")
    parser.add_argument("--review-source", default="v3_2_recall_fix_applied_manual_review")
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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_float(value: str) -> str:
    parsed = parse_float(value)
    if math.isnan(parsed):
        return ""
    return f"{parsed:.9g}"


def counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def score_distribution(rows: list[dict[str, str]], score_field: str) -> dict[str, float | None]:
    values = sorted(parse_float(row.get(score_field, "")) for row in rows)
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"min": None, "median": None, "p90": None, "p95": None, "max": None}

    def quantile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return {
        "min": values[0],
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def score_distribution_by(rows: list[dict[str, str]], group_field: str, score_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(group_field, "")].append(row)
    return {key: score_distribution(group, score_field) for key, group in sorted(grouped.items())}


def crosstab(rows: list[dict[str, str]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[row.get(left, "")][row.get(right, "")] += 1
    return {key: dict(value.most_common()) for key, value in sorted(table.items())}


def label_from_final(row: dict[str, str]) -> str:
    final_decision = row["final_manual_decision"].strip()
    final_downstream = row["final_downstream_decision"].strip()
    if final_decision == "cough":
        return "cough"
    if final_downstream in {"hard_negative", "clean_non_cough"}:
        return "non_cough"
    raise ValueError(
        f"Cannot map final fields to label for review_id={row.get('review_id')}: "
        f"final_manual_decision={final_decision} final_downstream_decision={final_downstream}"
    )


def negative_type(row: dict[str, str]) -> str:
    downstream = row["final_downstream_decision"].strip()
    if downstream == "hard_negative":
        return "hard_negative"
    if downstream == "clean_non_cough":
        return "clean_non_cough"
    return ""


def build_candidate_row(row: dict[str, str], source_domain: str, review_source: str) -> dict[str, str]:
    label = label_from_final(row)
    candidate_id = stable_id(
        "v3_2_recall_fix",
        row["review_id"],
        row["audio_file"],
        row["event_start_sec"],
        row["event_end_sec"],
        row["final_manual_decision"],
        row["final_candidate_role"],
    )
    notes = [
        "candidate_addon_only",
        "from_applied_manual_review",
        f"final_manual_decision={row['final_manual_decision']}",
        f"final_candidate_role={row['final_candidate_role']}",
    ]
    if row.get("person_id_for_analysis") == "p02":
        notes.append("p02_seen_person_sanity_not_generalization")
    if row.get("split_role_suggested") == "long_quiet_eval":
        notes.append("long_quiet_not_bulk_train")
    return {
        "sample_id": candidate_id,
        "clip_id": candidate_id,
        "source_audio_path": row["audio_file"],
        "source_json_path": row.get("metadata_json", ""),
        "event_start_sec": row["event_start_sec"],
        "event_end_sec": row["event_end_sec"],
        "label": label,
        "manual_subtype": row["final_manual_subtype"],
        "downstream_decision": row["final_downstream_decision"],
        "candidate_role": row["final_candidate_role"],
        "source_role": row["final_candidate_role"],
        "negative_type": negative_type(row),
        "person_id_for_analysis": row.get("person_id_for_analysis", ""),
        "source_actor_type": row.get("source_actor_type", ""),
        "scenario_canonical": row.get("scenario_canonical", ""),
        "split_role_suggested": row.get("split_role_suggested", ""),
        "source_domain": source_domain,
        "fpfix_cough_prob": format_float(row.get("fpfix_cough_prob", "")),
        "fpfix_non_cough_prob": format_float(row.get("fpfix_non_cough_prob", "")),
        "event_max_prob": format_float(row.get("event_max_prob", "")),
        "event_mean_prob": format_float(row.get("event_mean_prob", "")),
        "review_id": row["review_id"],
        "review_source": review_source,
        "final_manual_decision": row["final_manual_decision"],
        "final_label_included": row["final_label_included"],
        "recheck_found": row.get("recheck_found", ""),
        "recheck_applied": row.get("recheck_applied", ""),
        "notes": ";".join(notes),
    }


def table_lines(title: str, values: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| value | count |", "| --- | ---: |"]
    for key, value in values.items():
        lines.append(f"| {key} | {value} |")
    return lines


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Recall-Fix Candidate Addon Summary",
        "",
        "This stage builds a candidate label addon from the QA recheck applied review CSV only. "
        "It does not train, merge manifests, overwrite old labels, generate `audio_labels_v3_2_train.csv`, "
        "copy audio, modify raw WAV/JSON, or write `ml/data/raw/`.",
        "",
        "## Outputs",
        "",
        f"- candidate_csv: `{report['outputs']['candidate_csv']}`",
        f"- report_json: `{report['outputs']['report_json']}`",
        f"- summary_md: `{report['outputs']['summary_md']}`",
        "",
        "## Counts",
        "",
        f"- input_applied_rows: `{report['row_counts']['input_applied_rows']}`",
        f"- candidate_rows: `{report['row_counts']['candidate_rows']}`",
        f"- excluded_rows: `{report['row_counts']['excluded_rows']}`",
        "",
    ]
    lines.extend(table_lines("Label Distribution", report["label_distribution"]))
    lines.extend([""])
    lines.extend(table_lines("Candidate Role Distribution", report["candidate_role_distribution"]))
    lines.extend(["", "## Key Candidate Counts", ""])
    lines.append(f"- positive_recall_fix_candidate: `{report['key_counts']['positive_recall_fix_candidate']}`")
    lines.append(f"- hard_negative_candidate: `{report['key_counts']['hard_negative_candidate']}`")
    lines.extend(["", "## Validation", "", "| check | result |", "| --- | --- |"])
    for key, value in report["validation"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            f"Validation result: `{report['validation_result']}`",
            "",
            "## Notes",
            "",
            "- p02 is kept as seen-person sanity only, not generalization proof.",
            "- Long-quiet / clean non-cough material should not overwhelm hard negatives in later training.",
            "- Run candidate audit, dataset check, preprocess smoke, and person/split checks before any training or train-label merge.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    applied_path = resolve_path(args.applied_review)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    if not applied_path.exists():
        raise FileNotFoundError(applied_path)
    for path in (out_path, report_path, summary_path):
        ensure_new_file(path, args.overwrite)

    old_hash_before = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    forbidden_train_path = resolve_path("review/v3_2_recall_fix/audio_labels_v3_2_train.csv")
    applied_rows, applied_columns = read_csv(applied_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in applied_columns]
    if missing:
        raise ValueError(f"Applied review is missing required columns: {missing}")

    final_decision_counts = counts([row["final_manual_decision"] for row in applied_rows])
    final_downstream_counts = counts([row["final_downstream_decision"] for row in applied_rows])
    final_role_counts = counts([row["final_candidate_role"] for row in applied_rows])
    included_rows = [
        row
        for row in applied_rows
        if parse_bool(row["final_label_included"]) and row["final_manual_decision"] != "exclude"
    ]
    excluded_rows = [row for row in applied_rows if row not in included_rows]

    candidate_rows = [build_candidate_row(row, args.source_domain, args.review_source) for row in included_rows]
    candidate_columns = [
        "sample_id",
        "clip_id",
        "source_audio_path",
        "source_json_path",
        "event_start_sec",
        "event_end_sec",
        "label",
        "manual_subtype",
        "downstream_decision",
        "candidate_role",
        "source_role",
        "negative_type",
        "person_id_for_analysis",
        "source_actor_type",
        "scenario_canonical",
        "split_role_suggested",
        "source_domain",
        "fpfix_cough_prob",
        "fpfix_non_cough_prob",
        "event_max_prob",
        "event_mean_prob",
        "review_id",
        "review_source",
        "final_manual_decision",
        "final_label_included",
        "recheck_found",
        "recheck_applied",
        "notes",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_columns)
        writer.writeheader()
        writer.writerows(candidate_rows)

    old_hash_after = {path: sha256_or_missing(resolve_path(path)) for path in OLD_LABEL_PATHS}
    label_values = [row["label"] for row in candidate_rows]
    review_ids = [row["review_id"] for row in candidate_rows]
    sample_ids = [row["sample_id"] for row in candidate_rows]
    expected_candidate_count = sum(
        1
        for row in applied_rows
        if parse_bool(row["final_label_included"]) and row["final_manual_decision"] != "exclude"
    )
    score_values_unchanged = all(
        row["fpfix_cough_prob"] == format_float(source.get("fpfix_cough_prob", ""))
        and row["fpfix_non_cough_prob"] == format_float(source.get("fpfix_non_cough_prob", ""))
        for row, source in zip(candidate_rows, included_rows)
    )

    report: dict[str, Any] = {
        "stage": "v3.2-CANDIDATE-1",
        "inputs": {
            "applied_review_csv": str(applied_path),
        },
        "outputs": {
            "candidate_csv": str(out_path),
            "report_json": str(report_path),
            "summary_md": str(summary_path),
        },
        "reuse_check": {
            "existing_v3_1_builder": "scripts/build_v3_1_candidate_from_manual_reviews.py",
            "decision": "not reused directly",
            "reason": (
                "The v3.1 builder is tied to board_live_data_1/v3.1 metadata and old "
                "manual_decision/candidate_eligible fields; this v3.2 stage must consume final_* "
                "fields from the applied QA recheck CSV."
            ),
        },
        "row_counts": {
            "input_applied_rows": len(applied_rows),
            "candidate_rows": len(candidate_rows),
            "excluded_rows": len(excluded_rows),
            "expected_candidate_rows_from_final_fields": expected_candidate_count,
        },
        "final_decision_distribution": final_decision_counts,
        "final_downstream_distribution": final_downstream_counts,
        "final_candidate_role_distribution": final_role_counts,
        "label_distribution": counts(label_values),
        "candidate_role_distribution": counts([row["candidate_role"] for row in candidate_rows]),
        "person_id_for_analysis_distribution": counts([row["person_id_for_analysis"] for row in candidate_rows]),
        "source_actor_type_distribution": counts([row["source_actor_type"] for row in candidate_rows]),
        "scenario_distribution": counts([row["scenario_canonical"] for row in candidate_rows]),
        "split_role_distribution": counts([row["split_role_suggested"] for row in candidate_rows]),
        "downstream_decision_distribution": counts([row["downstream_decision"] for row in candidate_rows]),
        "source_domain_distribution": counts([row["source_domain"] for row in candidate_rows]),
        "fpfix_score_distribution_by_label": score_distribution_by(candidate_rows, "label", "fpfix_cough_prob"),
        "key_counts": {
            "positive_recall_fix_candidate": sum(
                1 for row in candidate_rows if row["candidate_role"] == "positive_recall_fix_candidate"
            ),
            "hard_negative_candidate": sum(
                1 for row in candidate_rows if row["candidate_role"] == "hard_negative_candidate"
            ),
            "positive_sanity_candidate": sum(
                1 for row in candidate_rows if row["candidate_role"] == "positive_sanity_candidate"
            ),
            "clean_non_cough_smoke_candidate": sum(
                1 for row in candidate_rows if row["candidate_role"] == "clean_non_cough_smoke_candidate"
            ),
        },
        "crosstabs": {
            "candidate_role_by_label": crosstab(candidate_rows, "candidate_role", "label"),
            "person_by_label": crosstab(candidate_rows, "person_id_for_analysis", "label"),
            "scenario_by_label": crosstab(candidate_rows, "scenario_canonical", "label"),
            "split_role_by_label": crosstab(candidate_rows, "split_role_suggested", "label"),
        },
        "old_label_hash_before": old_hash_before,
        "old_label_hash_after": old_hash_after,
        "score_values_unchanged": score_values_unchanged,
        "not_executed": NOT_EXECUTED,
    }

    validation = {
        "input_applied_rows_is_322": len(applied_rows) == 322,
        "candidate_rows_match_final_included_non_exclude": len(candidate_rows) == expected_candidate_count,
        "excluded_rows_is_3": len(excluded_rows) == 3,
        "final_decision_distribution_matches_expected": final_decision_counts == EXPECTED_FINAL_DECISION,
        "final_downstream_distribution_matches_expected": final_downstream_counts == EXPECTED_FINAL_DOWNSTREAM,
        "final_candidate_role_distribution_matches_expected": final_role_counts == EXPECTED_FINAL_ROLE,
        "every_candidate_has_label": all(row["label"] for row in candidate_rows),
        "labels_only_cough_or_non_cough": set(label_values).issubset({"cough", "non_cough"}),
        "review_id_unique": len(review_ids) == len(set(review_ids)),
        "sample_id_unique": len(sample_ids) == len(set(sample_ids)),
        "score_values_unchanged": score_values_unchanged,
        "old_label_hashes_unchanged": old_hash_before == old_hash_after,
        "audio_labels_v3_2_train_not_generated": not forbidden_train_path.exists(),
        "training_not_run": True,
        "manifest_merge_not_run": True,
        "ml_data_raw_not_written": True,
        "raw_wav_json_not_modified": True,
        "audio_not_copied": True,
    }
    report["validation"] = validation
    report["validation_result"] = "pass" if all(validation.values()) else "fail"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
