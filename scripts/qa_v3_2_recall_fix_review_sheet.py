from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DECISION_MAP = {
    "cough": "cough",
    "non_cough": "non_cough",
    "non-cough": "non_cough",
    "non cough": "non_cough",
    "noncough": "non_cough",
    "hard_negative": "non_cough",
    "hard negative": "non_cough",
    "clean_non_cough": "non_cough",
    "clean non cough": "non_cough",
    "exclude": "exclude",
    "uncertain": "exclude",
}

SUBTYPE_MAP = {
    "cough": "cough",
    "cough_train": "cough",
    "soft_cough": "soft_cough",
    "dry_cough": "dry_cough",
    "muffled_cough": "muffled_cough",
    "cough mixed with throat_clear": "cough_with_throat_clear",
    "cough_with_throat_clear": "cough_with_throat_clear",
    "background": "background",
    "quiet": "quiet",
    "speech": "speech",
    "throat_clear": "throat_clear",
    "sniff": "sniff",
    "knock": "knock",
    "handling_noise": "handling_noise",
    "impact": "impact",
    "uncertain": "uncertain",
    "exclude": "uncertain",
}

COUGH_SUBTYPES = {
    "cough",
    "soft_cough",
    "dry_cough",
    "muffled_cough",
    "cough_with_throat_clear",
}
NON_COUGH_SUBTYPES = {
    "background",
    "quiet",
    "speech",
    "throat_clear",
    "sniff",
    "knock",
    "handling_noise",
    "impact",
}
HARD_NEGATIVE_SCENARIOS = {
    "mixed_non_cough",
    "handling_noise",
    "knock",
    "impact",
    "speech",
    "throat_clear",
    "sniff",
}
BACKGROUND_SCENARIOS = {"quiet", "background"}
HARD_NEGATIVE_ACTORS = {
    "non_person_noise",
    "background_speech",
    "human_subject_negative",
}
REQUIRED_COLUMNS = [
    "review_id",
    "priority_bucket",
    "scenario_canonical",
    "source_actor_type",
    "person_id_for_analysis",
    "split_role_suggested",
    "fpfix_cough_prob",
    "manual_decision",
    "manual_subtype",
]
NOT_EXECUTED = [
    "training",
    "candidate build",
    "manifest merge",
    "audio_labels_v3_2_candidate_recall_fix.csv generation",
    "audio_labels_v3_2_train.csv generation",
    "raw WAV/JSON modification",
    "ml/data/raw write",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QA-normalize the v3.2 recall-fix manual review sheet without building labels."
    )
    parser.add_argument("--review-plan", required=True)
    parser.add_argument("--out", required=True, help="Normalized review CSV output path.")
    parser.add_argument("--report", required=True, help="Audit report JSON output path.")
    parser.add_argument("--summary-md", required=True, help="Audit summary Markdown output path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_new_file(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def norm_key(value: str) -> str:
    return norm_text(value).replace("-", "_").replace(" ", "_")


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_decision(raw: str) -> tuple[str, list[str]]:
    text = norm_text(raw)
    key = norm_key(raw)
    flags: list[str] = []
    if not text:
        return "unknown", ["blank_manual_decision"]
    if text in DECISION_MAP:
        canonical = DECISION_MAP[text]
    elif key in DECISION_MAP:
        canonical = DECISION_MAP[key]
    else:
        canonical = "unknown"
        flags.append("unknown_manual_decision")
    if text == "uncertain":
        flags.append("manual_decision_uncertain_mapped_to_exclude")
    if canonical == "unknown":
        flags.append("exclude_until_decision_fixed")
    return canonical, flags


def normalize_subtype(raw: str) -> tuple[str, str, list[str]]:
    text = norm_text(raw)
    key = norm_key(raw)
    flags: list[str] = []
    if not text:
        return "unknown", "unknown", ["blank_manual_subtype"]
    if text in SUBTYPE_MAP:
        subtype = SUBTYPE_MAP[text]
    elif key in SUBTYPE_MAP:
        subtype = SUBTYPE_MAP[key]
    else:
        subtype = "unknown"
        flags.append("unknown_manual_subtype")
    if subtype in COUGH_SUBTYPES:
        family = "cough"
    elif subtype in NON_COUGH_SUBTYPES:
        family = "non_cough"
    elif subtype == "uncertain":
        family = "uncertain"
    else:
        family = "unknown"
    return subtype, family, flags


def suggested_role(row: dict[str, str], decision: str, subtype_family: str, prob: float) -> tuple[str, str, list[str]]:
    scenario = row.get("scenario_canonical", "")
    actor = row.get("source_actor_type", "")
    split_role = row.get("split_role_suggested", "")
    bucket = row.get("priority_bucket", "")
    flags: list[str] = []

    cough_hint = scenario == "cough" and actor == "human_subject"
    non_cough_hint = scenario != "cough" or actor != "human_subject"

    if decision == "exclude":
        return "exclude", "exclude_until_re_review", flags
    if decision == "unknown":
        return "exclude", "exclude_until_decision_fixed", flags

    if decision == "cough":
        downstream = "cough"
        if non_cough_hint:
            role = "positive_boundary_review"
            flags.append("manual_cough_in_non_cough_hint")
        elif not math.isnan(prob) and prob < 0.50:
            role = "positive_recall_fix_candidate"
        else:
            role = "positive_sanity_candidate"
        if subtype_family == "non_cough":
            flags.append("cough_decision_with_non_cough_subtype")
        return downstream, role, flags

    # decision == non_cough
    if cough_hint:
        flags.append("manual_non_cough_in_cough_hint")
    if subtype_family == "cough":
        flags.append("non_cough_decision_with_cough_subtype")

    hard_negative_context = (
        scenario in HARD_NEGATIVE_SCENARIOS
        or actor in HARD_NEGATIVE_ACTORS
        or "hard_negative" in split_role
        or "hard_negative" in bucket
        or (not math.isnan(prob) and prob >= 0.50)
    )
    if hard_negative_context:
        return "hard_negative", "hard_negative_candidate", flags
    if scenario in BACKGROUND_SCENARIOS or actor == "background" or "quiet" in split_role or "smoke_negative" in split_role:
        return "clean_non_cough", "clean_non_cough_smoke_candidate", flags
    return "clean_non_cough", "non_cough_candidate", flags


def bool_text(value: bool) -> str:
    return "TRUE" if value else "FALSE"


def counts(values: list[str]) -> dict[str, int]:
    return dict(Counter(values).most_common())


def crosstab(rows: list[dict[str, str]], left: str, right: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[row.get(left, "")][row.get(right, "")] += 1
    return {key: dict(value.most_common()) for key, value in sorted(table.items())}


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# v3.2 Manual Review QA Normalization Summary",
        "",
        "This stage normalizes manual review fields into an audit-friendly copy. It does not build candidate labels, merge manifests, run training, modify raw WAV/JSON, or write `ml/data/raw/`.",
        "",
        "## Outputs",
        "",
        f"- normalized_csv: `{report['outputs']['normalized_csv']}`",
        f"- audit_report_json: `{report['outputs']['audit_report_json']}`",
        f"- summary_md: `{report['outputs']['summary_md']}`",
        "",
        "## Input",
        "",
        f"- review_plan: `{report['input']['review_plan']}`",
        f"- input_sha256_before: `{report['input']['sha256_before']}`",
        f"- input_sha256_after: `{report['input']['sha256_after']}`",
        f"- source_unchanged: `{report['validation']['source_review_plan_unchanged']}`",
        "",
        "## Decision Counts",
        "",
    ]
    for key, value in report["counts"]["manual_decision_canonical"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Downstream Decision Suggestions", ""])
    for key, value in report["counts"]["downstream_decision_suggested"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Candidate Role Suggestions", ""])
    for key, value in report["counts"]["candidate_role_suggested"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Subtype Counts", ""])
    for key, value in report["counts"]["manual_subtype_canonical"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## QA Flags", ""])
    if report["counts"]["qa_flags"]:
        for key, value in report["counts"]["qa_flags"].items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Validation", ""])
    for key, value in report["validation"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Boundary", ""])
    for item in NOT_EXECUTED:
        lines.append(f"- not executed: {item}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Inspect QA flags and confirm the decision/subtype mapping. Only after that should a separate candidate-builder step create `audio_labels_v3_2_candidate_recall_fix.csv` from reviewed rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.review_plan)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    for path in [out_path, report_path, summary_path]:
        ensure_new_file(path, args.overwrite)

    source_hash_before = sha256_file(input_path)
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        input_columns = list(reader.fieldnames)
        input_rows = [dict(row) for row in reader]

    normalized_rows: list[dict[str, str]] = []
    all_flags: list[str] = []
    for row in input_rows:
        decision_raw = row.get("manual_decision", "")
        subtype_raw = row.get("manual_subtype", "")
        decision, decision_flags = normalize_decision(decision_raw)
        subtype, subtype_family, subtype_flags = normalize_subtype(subtype_raw)
        prob = parse_float(row.get("fpfix_cough_prob", ""))
        downstream_decision, role, role_flags = suggested_role(row, decision, subtype_family, prob)

        flags = decision_flags + subtype_flags + role_flags
        if row.get("person_id_for_analysis") == "p02":
            flags.append("p02_seen_person_sanity")
        if row.get("source_actor_type") in {"background", "non_person_noise", "background_speech"}:
            flags.append("not_person_level_recall")
        if row.get("split_role_suggested") == "long_quiet_eval":
            flags.append("long_quiet_eval_not_bulk_train")

        needs_extra_review = any(
            flag
            in {
                "unknown_manual_decision",
                "unknown_manual_subtype",
                "manual_cough_in_non_cough_hint",
                "manual_non_cough_in_cough_hint",
                "cough_decision_with_non_cough_subtype",
                "non_cough_decision_with_cough_subtype",
                "manual_decision_uncertain_mapped_to_exclude",
            }
            for flag in flags
        )
        candidate_eligible = decision in {"cough", "non_cough"} and not any(
            flag in {"unknown_manual_decision", "unknown_manual_subtype"} for flag in flags
        )

        out_row = dict(row)
        out_row.update(
            {
                "manual_decision_raw": decision_raw,
                "manual_decision_canonical": decision,
                "manual_subtype_raw": subtype_raw,
                "manual_subtype_canonical": subtype,
                "manual_subtype_family": subtype_family,
                "downstream_decision_suggested": downstream_decision,
                "candidate_label_suggested": "non_cough" if downstream_decision in {"hard_negative", "clean_non_cough"} else downstream_decision,
                "candidate_role_suggested": role,
                "candidate_eligible_after_qa": bool_text(candidate_eligible),
                "needs_extra_review_before_candidate": bool_text(needs_extra_review),
                "qa_flags": ";".join(flags),
            }
        )
        normalized_rows.append(out_row)
        all_flags.extend(flags)

    output_columns = input_columns + [
        "manual_decision_raw",
        "manual_decision_canonical",
        "manual_subtype_raw",
        "manual_subtype_canonical",
        "manual_subtype_family",
        "downstream_decision_suggested",
        "candidate_label_suggested",
        "candidate_role_suggested",
        "candidate_eligible_after_qa",
        "needs_extra_review_before_candidate",
        "qa_flags",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)

    source_hash_after = sha256_file(input_path)
    report: dict[str, Any] = {
        "stage": "v3.2-SCAN-1C-QA-NORMALIZE",
        "input": {
            "review_plan": str(input_path),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
        },
        "outputs": {
            "normalized_csv": str(out_path),
            "audit_report_json": str(report_path),
            "summary_md": str(summary_path),
        },
        "rows": {
            "input": len(input_rows),
            "output": len(normalized_rows),
        },
        "counts": {
            "manual_decision_raw": counts([norm_text(row.get("manual_decision_raw", "")) for row in normalized_rows]),
            "manual_decision_canonical": counts([row["manual_decision_canonical"] for row in normalized_rows]),
            "manual_subtype_raw": counts([norm_text(row.get("manual_subtype_raw", "")) for row in normalized_rows]),
            "manual_subtype_canonical": counts([row["manual_subtype_canonical"] for row in normalized_rows]),
            "manual_subtype_family": counts([row["manual_subtype_family"] for row in normalized_rows]),
            "downstream_decision_suggested": counts([row["downstream_decision_suggested"] for row in normalized_rows]),
            "candidate_label_suggested": counts([row["candidate_label_suggested"] for row in normalized_rows]),
            "candidate_role_suggested": counts([row["candidate_role_suggested"] for row in normalized_rows]),
            "candidate_eligible_after_qa": counts([row["candidate_eligible_after_qa"] for row in normalized_rows]),
            "needs_extra_review_before_candidate": counts([row["needs_extra_review_before_candidate"] for row in normalized_rows]),
            "qa_flags": counts(all_flags),
        },
        "crosstabs": {
            "decision_by_bucket": crosstab(normalized_rows, "priority_bucket", "manual_decision_canonical"),
            "decision_by_scenario": crosstab(normalized_rows, "scenario_canonical", "manual_decision_canonical"),
            "downstream_decision_by_role": crosstab(
                normalized_rows, "candidate_role_suggested", "downstream_decision_suggested"
            ),
            "subtype_by_decision": crosstab(normalized_rows, "manual_decision_canonical", "manual_subtype_canonical"),
        },
        "normalization_policy": {
            "source_review_plan": "read-only; preserved by SHA-256 before/after check",
            "manual_decision_canonical": "cough/non_cough/exclude/unknown; uncertain maps to exclude with a QA flag",
            "manual_subtype_canonical": "conservative mapping; unknowns are flagged and not silently promoted",
            "downstream_decision_suggested": "cough/hard_negative/clean_non_cough/exclude for candidate-builder compatibility only",
            "candidate_label_suggested": "binary cough/non_cough/exclude suggestion, not a formal label CSV",
        },
        "not_executed": NOT_EXECUTED,
        "validation": {
            "row_count_preserved": len(input_rows) == len(normalized_rows),
            "source_review_plan_unchanged": source_hash_before == source_hash_after,
            "manual_decision_blank_count": sum(1 for row in normalized_rows if not norm_text(row.get("manual_decision_raw", ""))),
            "manual_subtype_blank_count": sum(1 for row in normalized_rows if not norm_text(row.get("manual_subtype_raw", ""))),
            "unknown_decision_count": sum(1 for row in normalized_rows if row["manual_decision_canonical"] == "unknown"),
            "unknown_subtype_count": sum(1 for row in normalized_rows if row["manual_subtype_canonical"] == "unknown"),
            "candidate_build_run": False,
            "training_run": False,
            "manifest_merge_run": False,
            "ml_data_raw_write": False,
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
