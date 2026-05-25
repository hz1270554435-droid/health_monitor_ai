from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


VALID_MANUAL_DECISIONS = {"", "cough", "hard_negative", "clean_non_cough", "uncertain", "exclude"}
MODEL_FALSE_ALARM_BUCKETS = {"hard_negative_review", "baseline_suspicious_review", "pseudo_cough_candidate"}

REQUIRED_REVIEW_COLUMNS = [
    "review_id",
    "review_file",
    "source_audio",
    "source_start_time",
    "source_end_time",
    "plan_bucket",
    "manual_decision",
    "baseline_cough_prob",
    "yamnet_top1",
    "yamnet_top1_score",
    "notes",
]

TRACE_COLUMNS = [
    "sample_weight",
    "source",
    "hard_negative_source",
    "review_batch",
    "review_id",
    "review_file",
    "source_audio",
    "source_start_time",
    "source_end_time",
    "derived_review_class",
    "baseline_cough_prob",
    "yamnet_top1",
    "yamnet_top1_score",
    "notes",
]

MERGED_EXTRA_COLUMNS = [
    "review_batch",
    "manual_decision_norm",
    "derived_review_class",
    "include_in_v3_candidate",
    "label",
    "hard_negative_source",
    "source_key",
    "duplicate_key",
    "manual_conflict",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge manual YAMNet review sheets and build audio_labels_v3_candidate.csv."
    )
    parser.add_argument("--base-labels", required=True)
    parser.add_argument(
        "--review-sheet",
        action="append",
        required=True,
        help="Review sheet CSV. Pass once per review batch.",
    )
    parser.add_argument("--out-merged", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--model-false-alarm-threshold", type=float, default=0.75)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def stable_hash(*parts: object, length: int = 12) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_clip_id(source_audio: str, start_time: float, end_time: float) -> str:
    return f"yamnet_review_{stable_hash(source_audio, f'{start_time:.6f}', f'{end_time:.6f}')}"


def stable_person_id(source_audio: str) -> str:
    return f"P_YAMNET_{stable_hash(source_audio, length=10)}"


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, keep_default_na=False, encoding=encoding), encoding
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}: {errors}")


def require_review_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_REVIEW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def row_key(audio_file: object, start_time: object, end_time: object) -> str:
    return f"{audio_file}|{float(start_time):.6f}|{float(end_time):.6f}"


def normalize_manual(value: object) -> str:
    return str(value).strip().lower()


def classify_row(row: pd.Series, threshold: float) -> tuple[str, bool, str, str, bool]:
    manual = str(row["manual_decision_norm"])
    bucket = str(row["plan_bucket"])
    baseline = float(row["baseline_cough_prob"])

    if manual == "":
        return "unreviewed", False, "", "", False
    if manual == "cough":
        return "cough", True, "cough", "", False
    if manual == "hard_negative":
        return "human_cough_like_hard_negative", True, "non_cough", "human_cough_like", True
    if manual == "clean_non_cough":
        if bucket in MODEL_FALSE_ALARM_BUCKETS and baseline >= threshold:
            return "model_false_alarm_hard_negative", True, "non_cough", "model_false_alarm", True
        if bucket == "pseudo_non_cough_candidate":
            return "clean_non_cough", True, "non_cough", "clean_non_cough", False
        if bucket == "uncertain_review":
            return "boundary_clean_non_cough", True, "non_cough", "boundary_clean_non_cough", False
        return "reviewed_clean_non_cough_other", False, "", "", False
    if manual == "exclude":
        return "exclude_low_quality", False, "", "", False
    if manual == "uncertain":
        return "holdout_uncertain", False, "", "", False
    return "invalid_manual_decision", False, "", "", False


def batch_name(path: Path, index: int) -> str:
    parent = path.parent.name
    if parent:
        return parent
    return f"review_batch_{index + 1}"


def load_review_sheets(paths: list[str], threshold: float) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    encodings: dict[str, str] = {}
    for index, raw_path in enumerate(paths):
        path = resolve_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Review sheet not found: {path}")
        frame, encoding = read_csv_with_fallback(path)
        encodings[str(path)] = encoding
        frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
        require_review_columns(frame, path)
        frame["review_batch"] = batch_name(path, index)
        frame["manual_decision_norm"] = frame["manual_decision"].map(normalize_manual)
        frame["source_start_time"] = frame["source_start_time"].astype(float)
        frame["source_end_time"] = frame["source_end_time"].astype(float)
        frame["baseline_cough_prob"] = frame["baseline_cough_prob"].astype(float)
        frame["source_key"] = frame["source_audio"].astype(str)
        frame["duplicate_key"] = frame.apply(
            lambda row: row_key(row["source_audio"], row["source_start_time"], row["source_end_time"]),
            axis=1,
        )
        derived = frame.apply(lambda row: classify_row(row, threshold), axis=1)
        frame["derived_review_class"] = [item[0] for item in derived]
        frame["include_in_v3_candidate"] = [bool(item[1]) for item in derived]
        frame["label"] = [item[2] for item in derived]
        frame["hard_negative_source"] = [item[3] for item in derived]
        frame["hard_negative"] = [bool(item[4]) for item in derived]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    invalid = sorted(set(merged["manual_decision_norm"]) - VALID_MANUAL_DECISIONS)
    if invalid:
        raise ValueError(f"Invalid manual_decision values: {invalid}")

    merged["manual_conflict"] = False
    reviewed = merged[merged["manual_decision_norm"] != ""]
    conflict_keys: set[str] = set()
    for key, group in reviewed.groupby("duplicate_key"):
        decisions = set(group["manual_decision_norm"])
        if len(decisions) > 1:
            conflict_keys.add(str(key))
    if conflict_keys:
        merged.loc[merged["duplicate_key"].isin(conflict_keys), "manual_conflict"] = True

    duplicate_seen: set[str] = set()
    duplicate_keep: list[bool] = []
    for _, row in merged.iterrows():
        key = str(row["duplicate_key"])
        if row["manual_decision_norm"] == "":
            duplicate_keep.append(True)
        elif key in duplicate_seen:
            duplicate_keep.append(False)
        else:
            duplicate_keep.append(True)
            duplicate_seen.add(key)
    merged["_first_duplicate"] = duplicate_keep
    merged.loc[~merged["_first_duplicate"], "include_in_v3_candidate"] = False
    merged.loc[merged["manual_conflict"], "include_in_v3_candidate"] = False
    return merged, encodings


def build_label_rows(base_labels: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    output_columns = list(base_labels.columns)
    for column in TRACE_COLUMNS:
        if column not in output_columns:
            output_columns.append(column)

    for column in output_columns:
        if column not in base_labels.columns:
            base_labels[column] = ""

    new_rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        source_audio = str(row["source_audio"])
        start_time = float(row["source_start_time"])
        end_time = float(row["source_end_time"])
        clip_id = stable_clip_id(source_audio, start_time, end_time)

        values = {column: "" for column in output_columns}
        values.update(
            {
                "clip_id": clip_id,
                "session_id": f"session_{clip_id}",
                "person_id": stable_person_id(source_audio),
                "start_time": start_time,
                "end_time": end_time,
                "label": row["label"],
                "audio_file": source_audio,
                "split": "train",
                "hard_negative": bool(row["hard_negative"]),
                "change_reason": str(row["derived_review_class"]),
                "sample_weight": 1.0,
                "source": "yamnet_review_gold",
                "hard_negative_source": row["hard_negative_source"],
                "review_batch": row["review_batch"],
                "review_id": row["review_id"],
                "review_file": row["review_file"],
                "source_audio": source_audio,
                "source_start_time": start_time,
                "source_end_time": end_time,
                "derived_review_class": row["derived_review_class"],
                "baseline_cough_prob": row["baseline_cough_prob"],
                "yamnet_top1": row["yamnet_top1"],
                "yamnet_top1_score": row["yamnet_top1_score"],
                "notes": row.get("notes", ""),
            }
        )
        new_rows.append(values)

    added = pd.DataFrame(new_rows, columns=output_columns)
    return pd.concat([base_labels[output_columns], added], ignore_index=True)


def counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# audio_labels_v3_candidate from YAMNet manual review",
        "",
        "This is a candidate label build only. It does not train a model and does not overwrite audio_labels_v2.csv.",
        "",
        "## Summary",
        "",
        f"- base_labels: `{report['base_labels']}`",
        f"- out_labels: `{report['out_labels']}`",
        f"- out_merged: `{report['out_merged']}`",
        f"- base_rows: {report['base_rows']}",
        f"- merged_rows: {report['merged_rows']}",
        f"- reviewed_rows: {report['reviewed_rows']}",
        f"- added_rows: {report['added_rows']}",
        f"- skipped_rows: {report['skipped_rows']}",
        f"- manual_conflict_rows: {report['manual_conflict_rows']}",
        f"- duplicate_skipped_rows: {report['duplicate_skipped_rows']}",
        f"- invalid_manual_decision_count: {report['invalid_manual_decision_count']}",
        "",
        "## Label Counts In Added Rows",
        "",
    ]
    for key, value in report["added_label_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Hard Negative Source Counts", ""])
    for key, value in report["added_hard_negative_source_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Derived Review Class Counts", ""])
    for key, value in report["derived_review_class_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Manual Decision Counts", ""])
    for key, value in report["manual_decision_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Source Audio Stats",
            "",
            f"- added_source_audio_count: {report['added_source_audio_count']}",
            f"- max_added_rows_per_source_audio: {report['max_added_rows_per_source_audio']}",
            "",
            "## Skipped Rows By Derived Class",
            "",
        ]
    )
    for key, value in report["skipped_derived_review_class_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_labels_path = resolve_path(args.base_labels)
    out_merged_path = resolve_path(args.out_merged)
    out_labels_path = resolve_path(args.out_labels)
    out_report_path = resolve_path(args.out_report)
    merged_report_json = out_merged_path.with_name("manual_review_merged_report.json")
    merged_report_md = out_merged_path.with_name("manual_review_merged_report.md")

    if not base_labels_path.exists():
        raise FileNotFoundError(f"Base labels not found: {base_labels_path}")
    if out_labels_path.resolve() == base_labels_path.resolve():
        raise ValueError("--out-labels must not overwrite --base-labels")
    for review_sheet in args.review_sheet:
        if not resolve_path(review_sheet).exists():
            raise FileNotFoundError(f"Review sheet not found: {resolve_path(review_sheet)}")

    base_labels = pd.read_csv(base_labels_path, keep_default_na=False)
    merged, encodings = load_review_sheets(args.review_sheet, float(args.model_false_alarm_threshold))
    ensure_dir(out_merged_path.parent)
    merged_output = merged.drop(columns=["_first_duplicate"], errors="ignore")
    merged_columns = [
        column
        for column in [*REQUIRED_REVIEW_COLUMNS, *MERGED_EXTRA_COLUMNS]
        if column in merged_output.columns
    ]
    remaining_columns = [column for column in merged_output.columns if column not in merged_columns]
    merged_output[[*merged_columns, *remaining_columns]].to_csv(out_merged_path, index=False, encoding="utf-8-sig")

    candidates = merged[merged["include_in_v3_candidate"]].copy()
    output_labels = build_label_rows(base_labels.copy(), candidates)
    ensure_dir(out_labels_path.parent)
    output_labels.to_csv(out_labels_path, index=False, encoding="utf-8")

    reviewed = merged[merged["manual_decision_norm"] != ""]
    skipped = merged[(merged["manual_decision_norm"] != "") & (~merged["include_in_v3_candidate"])]
    added_source_counts = candidates["source_audio"].value_counts() if not candidates.empty else pd.Series(dtype=int)
    report = {
        "base_labels": str(base_labels_path),
        "review_sheets": [str(resolve_path(path)) for path in args.review_sheet],
        "review_sheet_encodings": encodings,
        "out_merged": str(out_merged_path),
        "out_labels": str(out_labels_path),
        "out_report": str(out_report_path),
        "merged_report_json": str(merged_report_json),
        "merged_report_md": str(merged_report_md),
        "base_rows": int(len(base_labels)),
        "merged_rows": int(len(merged)),
        "reviewed_rows": int(len(reviewed)),
        "unreviewed_rows": int(len(merged) - len(reviewed)),
        "added_rows": int(len(candidates)),
        "output_rows": int(len(output_labels)),
        "skipped_rows": int(len(skipped)),
        "manual_conflict_rows": int(merged["manual_conflict"].sum()),
        "duplicate_skipped_rows": int(((~merged["_first_duplicate"]) & (merged["manual_decision_norm"] != "")).sum()),
        "invalid_manual_decision_count": 0,
        "model_false_alarm_threshold": float(args.model_false_alarm_threshold),
        "manual_decision_counts": counts(merged["manual_decision_norm"].replace("", "<blank>")),
        "derived_review_class_counts": counts(merged["derived_review_class"]),
        "reviewed_derived_review_class_counts": counts(reviewed["derived_review_class"]),
        "added_label_counts": counts(candidates["label"]) if not candidates.empty else {},
        "added_hard_negative_source_counts": counts(candidates["hard_negative_source"]) if not candidates.empty else {},
        "skipped_derived_review_class_counts": counts(skipped["derived_review_class"]) if not skipped.empty else {},
        "added_source_audio_count": int(candidates["source_audio"].nunique()) if not candidates.empty else 0,
        "max_added_rows_per_source_audio": int(added_source_counts.max()) if not added_source_counts.empty else 0,
    }
    ensure_dir(merged_report_json.parent)
    merged_report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(merged_report_md, report)
    ensure_dir(out_report_path.parent)
    write_markdown_report(out_report_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
