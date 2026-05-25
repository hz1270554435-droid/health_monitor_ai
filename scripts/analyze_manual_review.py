from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


MODEL_FALSE_ALARM_BUCKETS = {"hard_negative_review", "baseline_suspicious_review"}
VALID_MANUAL_DECISIONS = {"", "cough", "hard_negative", "clean_non_cough", "uncertain", "exclude"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze human review decisions from YAMNet hard-mining review_sheet.csv."
    )
    parser.add_argument(
        "--review-sheet",
        default="data/human_review/第一次人工重听/review_sheet.csv",
    )
    parser.add_argument("--out-dir", default="review/yamnet_full_non_cough_v1")
    parser.add_argument("--model-false-alarm-threshold", type=float, default=0.75)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = [
        "review_file",
        "source_audio",
        "source_start_time",
        "source_end_time",
        "plan_bucket",
        "manual_decision",
        "baseline_cough_prob",
        "yamnet_top1",
        "notes",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def derived_review_class(row: pd.Series, threshold: float) -> str:
    decision = str(row["manual_decision"]).strip()
    bucket = str(row["plan_bucket"]).strip()
    baseline = float(row["baseline_cough_prob"])
    if decision == "":
        return "unreviewed"
    if decision == "cough":
        return "cough"
    if decision == "hard_negative":
        return "human_cough_like_hard_negative"
    if decision == "clean_non_cough":
        if bucket in MODEL_FALSE_ALARM_BUCKETS and baseline >= threshold:
            return "model_false_alarm_hard_negative"
        if bucket == "pseudo_non_cough_candidate":
            return "clean_non_cough"
        return "reviewed_clean_non_cough_other"
    if decision == "exclude":
        return "exclude_low_quality"
    if decision == "uncertain":
        return "holdout_uncertain"
    return "invalid_manual_decision"


def counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def table_lines(title: str, data: dict[str, int]) -> list[str]:
    lines = [f"## {title}", ""]
    if not data:
        lines.append("_None._")
    else:
        lines.extend(f"- {key}: {value}" for key, value in data.items())
    lines.append("")
    return lines


def pct_table(df: pd.DataFrame, index: str, columns: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    ct = pd.crosstab(df[index], df[columns])
    pct = ct.div(ct.sum(axis=1), axis=0).mul(100).round(2)
    rows: list[dict[str, Any]] = []
    for idx in ct.index:
        row: dict[str, Any] = {"bucket": str(idx), "total": int(ct.loc[idx].sum())}
        for col in ct.columns:
            row[f"{col}_count"] = int(ct.loc[idx, col])
            row[f"{col}_pct"] = float(pct.loc[idx, col])
        rows.append(row)
    return rows


def markdown_crosstab(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_None._\n"
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = ["bucket", "total"]
    columns = preferred + [col for col in columns if col not in preferred]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Manual review analysis - first listening batch",
        "",
        "This report only analyzes human review decisions. It does not modify labels, export clips, or train a model.",
        "",
        "## Summary",
        "",
        f"- review_sheet: `{report['review_sheet']}`",
        f"- total_rows: {report['total_rows']}",
        f"- reviewed_rows: {report['reviewed_rows']}",
        f"- unreviewed_rows: {report['unreviewed_rows']}",
        f"- reviewed_sources: {report['reviewed_sources']}",
        f"- model_false_alarm_threshold: {report['model_false_alarm_threshold']}",
        f"- invalid_manual_decisions: {report['invalid_manual_decisions']}",
        "",
    ]
    lines.extend(table_lines("Manual decision counts", report["manual_decision_counts"]))
    lines.extend(table_lines("Derived review class counts", report["derived_review_class_counts"]))

    lines.extend(["## Bucket vs manual decision", "", markdown_crosstab(report["bucket_manual_table"]), ""])
    lines.extend(["## Bucket vs derived review class", "", markdown_crosstab(report["bucket_derived_table"]), ""])

    lines.extend(table_lines("Top YAMNet top1 labels in reviewed rows", report["reviewed_yamnet_top1_counts"]))
    lines.extend(table_lines("Top YAMNet top1 labels for model false alarms", report["model_false_alarm_top1_counts"]))
    lines.extend(table_lines("Top YAMNet top1 labels for human cough-like hard negatives", report["human_hard_negative_top1_counts"]))
    lines.extend(table_lines("Notes counts", report["notes_counts"]))

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `model_false_alarm_hard_negative` rows are confirmed non-cough clips that the baseline scored as cough-like.",
            "- `human_cough_like_hard_negative` rows are non-cough clips that also sounded cough-like to the reviewer.",
            "- `exclude_low_quality` rows should not enter cough/non_cough training.",
            "- `reviewed_clean_non_cough_other` rows are reviewed clean non-cough clips outside the current model-false-alarm policy.",
            "",
            "## Next recommended step",
            "",
            "Generate a second, lower-duplication review plan with `max_per_source=1`, `min_start_gap_sec=5.0`, reduced baseline suspicious quota, and no pseudo cough quota.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_sheet = resolve_path(args.review_sheet)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    out_json = out_dir / "manual_review_analysis.json"
    out_md = out_dir / "manual_review_analysis.md"

    if not review_sheet.exists():
        raise FileNotFoundError(f"Review sheet not found: {review_sheet}")

    df = pd.read_csv(review_sheet, keep_default_na=False)
    require_columns(df, review_sheet)
    df["manual_decision"] = df["manual_decision"].astype(str).str.strip()
    invalid_manual = sorted(set(df["manual_decision"]) - VALID_MANUAL_DECISIONS)
    df["derived_review_class"] = df.apply(
        lambda row: derived_review_class(row, float(args.model_false_alarm_threshold)),
        axis=1,
    )

    reviewed = df[df["manual_decision"] != ""].copy()
    model_false_alarm = reviewed[reviewed["derived_review_class"] == "model_false_alarm_hard_negative"]
    human_hard_negative = reviewed[reviewed["derived_review_class"] == "human_cough_like_hard_negative"]

    source_counts = reviewed["source_audio"].value_counts() if not reviewed.empty else pd.Series(dtype=int)
    report = {
        "review_sheet": str(review_sheet),
        "manual_review_analysis_json": str(out_json),
        "manual_review_analysis_md": str(out_md),
        "total_rows": int(len(df)),
        "reviewed_rows": int(len(reviewed)),
        "unreviewed_rows": int(len(df) - len(reviewed)),
        "reviewed_sources": int(reviewed["source_audio"].nunique()) if not reviewed.empty else 0,
        "max_reviewed_per_source": int(source_counts.max()) if not source_counts.empty else 0,
        "sources_with_multiple_reviewed": int((source_counts > 1).sum()) if not source_counts.empty else 0,
        "model_false_alarm_threshold": float(args.model_false_alarm_threshold),
        "invalid_manual_decisions": invalid_manual,
        "manual_decision_counts": counts(df["manual_decision"].replace("", "<blank>")),
        "reviewed_manual_decision_counts": counts(reviewed["manual_decision"]) if not reviewed.empty else {},
        "derived_review_class_counts": counts(df["derived_review_class"]),
        "reviewed_derived_review_class_counts": counts(reviewed["derived_review_class"]) if not reviewed.empty else {},
        "bucket_manual_table": pct_table(reviewed, "plan_bucket", "manual_decision"),
        "bucket_derived_table": pct_table(reviewed, "plan_bucket", "derived_review_class"),
        "reviewed_yamnet_top1_counts": counts(reviewed["yamnet_top1"].head(0))
        if reviewed.empty
        else {str(k): int(v) for k, v in reviewed["yamnet_top1"].value_counts().head(30).items()},
        "model_false_alarm_top1_counts": {}
        if model_false_alarm.empty
        else {str(k): int(v) for k, v in model_false_alarm["yamnet_top1"].value_counts().head(30).items()},
        "human_hard_negative_top1_counts": {}
        if human_hard_negative.empty
        else {str(k): int(v) for k, v in human_hard_negative["yamnet_top1"].value_counts().head(30).items()},
        "notes_counts": {str(k): int(v) for k, v in reviewed["notes"].replace("", "<blank>").value_counts().head(30).items()}
        if not reviewed.empty
        else {},
        "v3_candidate_counts_if_policy_applied": {
            "cough": int((reviewed["derived_review_class"] == "cough").sum()),
            "human_cough_like_hard_negative": int((reviewed["derived_review_class"] == "human_cough_like_hard_negative").sum()),
            "model_false_alarm_hard_negative": int((reviewed["derived_review_class"] == "model_false_alarm_hard_negative").sum()),
            "clean_non_cough": int((reviewed["derived_review_class"] == "clean_non_cough").sum()),
            "excluded_from_training": int(
                reviewed["derived_review_class"].isin(
                    ["exclude_low_quality", "holdout_uncertain", "reviewed_clean_non_cough_other"]
                ).sum()
            ),
        },
    }

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(out_md, report)
    print(json.dumps(report["v3_candidate_counts_if_policy_applied"], indent=2, ensure_ascii=False))
    print(f"OK: wrote {out_md} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
