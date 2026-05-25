from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover - fallback is for minimal environments.
    yaml = None


ALLOWED_MANUAL_DECISIONS = ["cough", "hard_negative", "clean_non_cough", "uncertain", "exclude"]
ELIGIBLE_DECISIONS = {"cough", "hard_negative", "clean_non_cough"}
EXCLUDE_DECISIONS = {"uncertain", "exclude"}
NORMALIZED_SUBTYPES = [
    "speech",
    "loud_speech",
    "silence",
    "breath",
    "throat_clear",
    "cough_with_throat_clear",
    "knock",
    "handling_noise",
    "cough_with_speech",
    "laugh",
    "nasal_noise",
    "clear_cough",
    "unknown",
]
NOT_EXECUTED = [
    "build_v3_1_candidate_from_manual_reviews.py",
    "audio_labels_v3_1_candidate.csv generation",
    "training",
    "audio copy",
    "ml/data/raw modification",
]

STATIC_MAPPING: dict[str, str] = {
    "讲话": "speech",
    "讲话加脚步声": "speech",
    "讲话加关门声": "speech",
    "讲话加撞击": "speech",
    "讲话加打哈欠": "speech",
    "讲话加深呼吸": "speech",
    "讲话加突然喊叫": "loud_speech",
    "安静": "silence",
    "深呼吸": "breath",
    "深呼气": "breath",
    "呼气加撞击": "breath",
    "深呼气加撞击": "breath",
    "撞击": "knock",
    "敲击": "knock",
    "轻微敲击声": "knock",
    "撞击加刮蹭": "handling_noise",
    "咳嗽加讲话": "cough_with_speech",
    "讲话加咳嗽": "cough_with_speech",
    "讲话加咯咯笑": "laugh",
    "笑声": "laugh",
    "清鼻子": "nasal_noise",
}

DECISION_AWARE_MAPPING: dict[str, dict[str, str]] = {
    "大叫加轻咳": {
        "cough": "cough_with_speech",
        "hard_negative": "loud_speech",
        "clean_non_cough": "loud_speech",
        "uncertain": "loud_speech",
        "exclude": "loud_speech",
    },
    "清嗓子": {
        "cough": "cough_with_throat_clear",
        "hard_negative": "throat_clear",
        "clean_non_cough": "throat_clear",
        "uncertain": "throat_clear",
        "exclude": "throat_clear",
    },
    "咳嗽，清嗓": {
        "cough": "cough_with_throat_clear",
        "hard_negative": "throat_clear",
        "clean_non_cough": "throat_clear",
        "uncertain": "throat_clear",
        "exclude": "throat_clear",
    },
    "轻咳加清嗓": {
        "cough": "cough_with_throat_clear",
        "hard_negative": "throat_clear",
        "clean_non_cough": "throat_clear",
        "uncertain": "throat_clear",
        "exclude": "throat_clear",
    },
    "咳嗽加清嗓子": {
        "cough": "cough_with_throat_clear",
        "hard_negative": "throat_clear",
        "clean_non_cough": "throat_clear",
        "uncertain": "throat_clear",
        "exclude": "throat_clear",
    },
    "轻声咳嗽": {
        "cough": "clear_cough",
        "hard_negative": "unknown",
        "clean_non_cough": "unknown",
        "uncertain": "unknown",
        "exclude": "unknown",
    },
}

UNKNOWN_RAW_REASONS: dict[str, str] = {
    "": "blank subtype text",
    "打哈欠": "yawn is not in the current normalized subtype vocabulary",
    "咳嗽加呛到": "choking-like cough text needs reviewer-specific interpretation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA and normalize filled v3.1 manual review sheets.")
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mapping-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalized_subtype(raw: str, decision: str) -> str:
    if raw in STATIC_MAPPING:
        return STATIC_MAPPING[raw]
    if raw in DECISION_AWARE_MAPPING:
        return DECISION_AWARE_MAPPING[raw].get(decision, "unknown")
    return "unknown"


def crosstab(frame: pd.DataFrame, row: str, column: str) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame[row], frame[column], dropna=False)
    return {str(index): {str(key): int(value) for key, value in values.items()} for index, values in table.iterrows()}


def counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def mapping_payload(raw_values: list[str]) -> dict[str, Any]:
    raw_mapping: dict[str, Any] = {}
    for raw in raw_values:
        if raw in STATIC_MAPPING:
            raw_mapping[raw] = {"type": "static", "norm": STATIC_MAPPING[raw]}
        elif raw in DECISION_AWARE_MAPPING:
            raw_mapping[raw] = {"type": "decision_aware", "norm_by_manual_decision": DECISION_AWARE_MAPPING[raw]}
        else:
            raw_mapping[raw] = {"type": "unknown", "norm": "unknown", "reason": UNKNOWN_RAW_REASONS.get(raw, "unmapped")}
    return {
        "stage": "v3.1-REVIEWQA-1",
        "allowed_manual_decisions": ALLOWED_MANUAL_DECISIONS,
        "normalized_subtypes": NORMALIZED_SUBTYPES,
        "policy": {
            "manual_subtype_raw": "preserve original manual_subtype text without rewriting",
            "manual_subtype_norm": "minimal conservative mapping from Chinese free text",
            "candidate_eligible": "true only for cough, hard_negative, clean_non_cough",
            "candidate_exclude": "true only for uncertain, exclude",
        },
        "raw_subtype_mapping": raw_mapping,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage v3.1-REVIEWQA-1 Summary",
        "",
        "This stage audits the filled manual review sheet and normalizes subtype text. It does not build "
        "candidate labels, train a model, copy audio, or modify ml/data/raw/.",
        "",
        "## Outputs",
        "",
        f"- cleaned_csv: `{report['cleaned_csv']}`",
        f"- subtype_mapping_yaml: `{report['subtype_mapping_yaml']}`",
        f"- audit_report_json: `{report['audit_report_json']}`",
        "",
        "## Manual Decisions",
        "",
    ]
    for key, value in report["manual_decision_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Candidate Flags",
            "",
            f"- candidate_eligible_count: `{report['candidate_eligible_count']}`",
            f"- candidate_exclude_count: `{report['candidate_exclude_count']}`",
            "",
            "## Person Distribution",
            "",
        ]
    )
    for key, value in report["person_id_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Normalized Subtypes", ""])
    for key, value in report["manual_subtype_norm_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Unknown Raw Subtypes", ""])
    if report["unknown_raw_subtype_counts"]:
        for key, value in report["unknown_raw_subtype_counts"].items():
            rendered = "<blank>" if key == "" else key
            reason = report["unknown_raw_subtype_reasons"].get(key, "unmapped")
            lines.append(f"- {rendered}: `{value}` ({reason})")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
            "- This batch is all `person_id=p01`; do not use it alone as a generalization evaluation set.",
            "",
            "## Not Executed",
            "",
        ]
    )
    for item in report["not_executed"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_sheet = resolve_path(args.review_sheet)
    out_path = resolve_path(args.out)
    mapping_path = resolve_path(args.mapping_out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    if not review_sheet.exists():
        raise FileNotFoundError(f"Review sheet not found: {review_sheet}")
    for path in (out_path, mapping_path, report_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")

    frame = pd.read_csv(review_sheet, keep_default_na=False)
    required = {"manual_decision", "manual_subtype", "person_id", "review_bucket", "yamnet_top1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Review sheet is missing required columns: {missing}")

    cleaned = frame.copy()
    cleaned["manual_decision"] = cleaned["manual_decision"].astype(str).str.strip()
    cleaned["manual_subtype_raw"] = cleaned["manual_subtype"].astype(str).str.strip()
    blank_decisions = int((cleaned["manual_decision"] == "").sum())
    if blank_decisions:
        raise ValueError(f"manual_decision has blank values: {blank_decisions}")
    invalid_decisions = sorted(set(cleaned["manual_decision"]) - set(ALLOWED_MANUAL_DECISIONS))
    if invalid_decisions:
        raise ValueError(f"manual_decision has invalid values: {invalid_decisions}")

    cleaned["manual_subtype_norm"] = [
        normalized_subtype(raw, decision)
        for raw, decision in zip(cleaned["manual_subtype_raw"], cleaned["manual_decision"], strict=True)
    ]
    cleaned["candidate_eligible"] = cleaned["manual_decision"].isin(ELIGIBLE_DECISIONS)
    cleaned["candidate_exclude"] = cleaned["manual_decision"].isin(EXCLUDE_DECISIONS)

    if int((cleaned["manual_subtype_norm"].astype(str).str.strip() == "").sum()):
        raise ValueError("manual_subtype_norm must be non-empty for every row")

    raw_values = sorted(cleaned["manual_subtype_raw"].unique().tolist())
    mapping = mapping_payload(raw_values)
    unknown_rows = cleaned[cleaned["manual_subtype_norm"] == "unknown"].copy()
    unknown_counts = counts(unknown_rows["manual_subtype_raw"]) if not unknown_rows.empty else {}
    report = {
        "stage": "v3.1-REVIEWQA-1",
        "input_review_sheet": str(review_sheet),
        "cleaned_csv": str(out_path),
        "subtype_mapping_yaml": str(mapping_path),
        "audit_report_json": str(report_path),
        "summary_md": str(summary_path),
        "total_rows": int(len(cleaned)),
        "manual_decision_counts": counts(cleaned["manual_decision"]),
        "manual_decision_invalid_values": invalid_decisions,
        "manual_decision_blank_count": blank_decisions,
        "manual_subtype_raw_counts": counts(cleaned["manual_subtype_raw"].replace("", "<blank>")),
        "manual_subtype_norm_counts": counts(cleaned["manual_subtype_norm"]),
        "unknown_raw_subtype_counts": unknown_counts,
        "unknown_raw_subtype_reasons": {
            raw: UNKNOWN_RAW_REASONS.get(raw, "unmapped") for raw in unknown_counts.keys()
        },
        "candidate_eligible_count": int(cleaned["candidate_eligible"].sum()),
        "candidate_exclude_count": int(cleaned["candidate_exclude"].sum()),
        "person_id_counts": counts(cleaned["person_id"]),
        "all_person_id_p01": set(cleaned["person_id"].astype(str)) == {"p01"},
        "person_id_warning": "All rows are person_id=p01; do not use this batch alone as a generalization evaluation set.",
        "crosstabs": {
            "manual_decision_by_manual_subtype_raw": crosstab(cleaned, "manual_decision", "manual_subtype_raw"),
            "manual_decision_by_manual_subtype_norm": crosstab(cleaned, "manual_decision", "manual_subtype_norm"),
            "review_bucket_by_manual_decision": crosstab(cleaned, "review_bucket", "manual_decision"),
            "yamnet_top1_by_manual_decision": crosstab(cleaned, "yamnet_top1", "manual_decision"),
        },
        "not_executed": NOT_EXECUTED,
    }

    ensure_parent(out_path)
    cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")
    write_yaml(mapping_path, mapping)
    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ensure_parent(summary_path)
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
