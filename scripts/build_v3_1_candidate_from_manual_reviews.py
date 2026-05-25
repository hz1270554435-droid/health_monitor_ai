from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import row_key, safe_token, stable_id


VALID_MANUAL_DECISIONS = {"", "cough", "hard_negative", "clean_non_cough", "uncertain", "exclude"}
ADDON_ELIGIBLE_DECISIONS = {"cough", "hard_negative", "clean_non_cough"}
ADDON_EXCLUDE_DECISIONS = {"uncertain", "exclude"}
ADDON_NOT_EXECUTED = [
    "training",
    "raw-data modification",
    "audio copy",
    "final label merge",
    "audio_labels_v3.csv overwrite",
    "YAMNet/v3 score promotion to formal labels",
]
ADDON_REQUIRED_COLUMNS = [
    "source_domain",
    "person_id",
    "session_id",
    "original_path",
    "source_audio",
    "window_start_sec",
    "window_end_sec",
    "clip_path",
    "review_bucket",
    "manual_decision",
    "manual_subtype_raw",
    "manual_subtype_norm",
    "v3_cough_prob",
    "yamnet_top1",
    "yamnet_cough_score",
    "yamnet_speech_score",
    "yamnet_throat_like_score",
    "candidate_eligible",
    "candidate_exclude",
]
ADDON_OUTPUT_COLUMNS = [
    "candidate_id",
    "clip_id",
    "label",
    "negative_type",
    "source_domain",
    "person_id",
    "session_id",
    "original_path",
    "source_audio",
    "canonical_audio_path",
    "window_start_sec",
    "window_end_sec",
    "clip_path",
    "review_bucket",
    "manual_decision",
    "manual_subtype_raw",
    "manual_subtype_norm",
    "v3_cough_prob",
    "yamnet_top1",
    "yamnet_cough_score",
    "yamnet_speech_score",
    "yamnet_throat_like_score",
    "candidate_eligible",
    "candidate_exclude",
    "review_clip_is_training_source",
    "canonical_training_window",
]
REQUIRED_REVIEW_COLUMNS = [
    "clip_id",
    "clip_path",
    "source_audio",
    "source_domain",
    "session_id",
    "person_id",
    "scenario_hint",
    "start_time",
    "end_time",
    "yamnet_top1",
    "yamnet_top5",
    "yamnet_cough_score",
    "v3_cough_prob",
    "v3_decision_075",
    "v3_decision_095",
    "review_bucket",
    "review_priority",
    "manual_decision",
    "manual_subtype",
    "manual_notes",
]

TRACE_COLUMNS = [
    "sample_weight",
    "source",
    "source_domain",
    "scenario_hint",
    "hard_negative_source",
    "manual_subtype",
    "review_bucket",
    "review_priority",
    "review_id",
    "review_file",
    "source_audio",
    "source_start_time",
    "source_end_time",
    "derived_review_class",
    "v3_cough_prob",
    "v3_decision_075",
    "v3_decision_095",
    "yamnet_top1",
    "yamnet_top5",
    "yamnet_cough_score",
    "opera_cough_score",
    "opera_hard_negative_score",
    "opera_rank",
    "manual_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audio_labels_v3_1_candidate.csv from v3.1 review sheets.")
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--base-labels", default="data/labels/audio_labels_v3.csv")
    parser.add_argument(
        "--review-sheet",
        action="append",
        default=None,
        help="Review sheet CSV. Defaults to review_dir/review_sheet.csv from config.",
    )
    parser.add_argument("--out-labels", default=None)
    parser.add_argument("--out-sessions", default=None)
    parser.add_argument("--out-merged", default=None)
    parser.add_argument("--out-report", default=None)
    parser.add_argument("--out-report-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--addon-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_csv_with_fallback(path: Path) -> tuple[pd.DataFrame, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, keep_default_na=False, encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode {path}")


def require_review_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_REVIEW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_manual(value: object) -> str:
    return str(value).strip().lower()


def normalize_subtype(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    return safe_token(text.replace("-", "_").replace(" ", "_"), max_len=64).lower()


def source_with_subtype(prefix: str, subtype: str) -> str:
    return f"{prefix}_{subtype}" if subtype else prefix


def classify_row(row: pd.Series) -> tuple[str, bool, str, bool, str]:
    manual = str(row["manual_decision_norm"])
    bucket = str(row["review_bucket"])
    subtype = str(row["manual_subtype_norm"])

    if manual == "":
        return "unreviewed", False, "", False, ""
    if manual == "uncertain":
        return "holdout_uncertain", False, "", False, ""
    if manual == "exclude":
        return "exclude_low_quality", False, "", False, ""
    if manual == "cough":
        return "cough", True, "cough", False, ""
    if manual == "hard_negative":
        return (
            "board_human_cough_like_hard_negative",
            True,
            "non_cough",
            True,
            source_with_subtype("board_human_cough_like", subtype),
        )
    if manual == "clean_non_cough":
        if bucket == "high_confidence_board_false_alarm_review":
            return (
                "board_high_conf_false_alarm_hard_negative",
                True,
                "non_cough",
                True,
                source_with_subtype("board_high_conf_false_alarm", subtype),
            )
        if bucket == "board_false_alarm_review":
            return (
                "board_false_alarm_hard_negative",
                True,
                "non_cough",
                True,
                source_with_subtype("board_false_alarm", subtype),
            )
        if bucket == "board_uncertain_review":
            return "board_boundary_clean_non_cough", True, "non_cough", False, "board_boundary_clean_non_cough"
        if bucket == "board_clean_non_cough_candidate":
            return "board_clean_non_cough", True, "non_cough", False, "board_clean_non_cough"
        if bucket == "board_cough_candidate_review":
            return "board_cough_candidate_clean_non_cough", True, "non_cough", False, "board_cough_candidate_clean_non_cough"
        return "board_reviewed_clean_non_cough", True, "non_cough", False, "board_reviewed_clean_non_cough"
    return "invalid_manual_decision", False, "", False, ""


def batch_name(path: Path, index: int) -> str:
    return path.parent.name or f"v3_1_review_batch_{index + 1}"


def load_review_sheets(paths: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
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
        invalid = sorted(set(frame["manual_decision_norm"]) - VALID_MANUAL_DECISIONS)
        if invalid:
            raise ValueError(f"{path} has invalid manual_decision values: {invalid}")
        frame["manual_subtype_norm"] = frame["manual_subtype"].map(normalize_subtype)
        frame["start_time"] = frame["start_time"].astype(float)
        frame["end_time"] = frame["end_time"].astype(float)
        frame["duplicate_key"] = frame.apply(
            lambda row: row_key(row["source_audio"], row["start_time"], row["end_time"]),
            axis=1,
        )
        derived = frame.apply(classify_row, axis=1)
        frame["derived_review_class"] = [item[0] for item in derived]
        frame["include_in_v3_1_candidate"] = [bool(item[1]) for item in derived]
        frame["label"] = [item[2] for item in derived]
        frame["hard_negative"] = [bool(item[3]) for item in derived]
        frame["hard_negative_source"] = [item[4] for item in derived]
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged["manual_conflict"] = False
    reviewed = merged[merged["manual_decision_norm"] != ""]
    conflict_keys: set[str] = set()
    for key, group in reviewed.groupby("duplicate_key"):
        decisions = set(group["manual_decision_norm"])
        if len(decisions) > 1:
            conflict_keys.add(str(key))
    if conflict_keys:
        merged.loc[merged["duplicate_key"].isin(conflict_keys), "manual_conflict"] = True

    seen: set[str] = set()
    keep_first: list[bool] = []
    for _, row in merged.iterrows():
        key = str(row["duplicate_key"])
        if row["manual_decision_norm"] == "":
            keep_first.append(True)
        elif key in seen:
            keep_first.append(False)
        else:
            keep_first.append(True)
            seen.add(key)
    merged["_first_duplicate"] = keep_first
    merged.loc[~merged["_first_duplicate"], "include_in_v3_1_candidate"] = False
    merged.loc[merged["manual_conflict"], "include_in_v3_1_candidate"] = False
    return merged, encodings


def output_columns(base_labels: pd.DataFrame) -> list[str]:
    columns = list(base_labels.columns)
    for column in TRACE_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def build_label_rows(base_labels: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    columns = output_columns(base_labels)
    for column in columns:
        if column not in base_labels.columns:
            base_labels[column] = ""

    new_rows: list[dict[str, object]] = []
    for _, row in candidates.iterrows():
        clip_id = str(row["clip_id"]) or stable_id("v3_1_review", row["source_audio"], row["start_time"], row["end_time"])
        values = {column: "" for column in columns}
        values.update(
            {
                "clip_id": clip_id,
                "session_id": row["session_id"],
                "person_id": row["person_id"],
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "label": row["label"],
                "audio_file": row["source_audio"],
                "split": "train",
                "hard_negative": bool(row["hard_negative"]),
                "change_reason": row["derived_review_class"],
                "sample_weight": 1.0,
                "source": "v3_1_manual_review",
                "source_domain": row["source_domain"],
                "scenario_hint": row["scenario_hint"],
                "hard_negative_source": row["hard_negative_source"],
                "manual_subtype": row["manual_subtype_norm"],
                "review_bucket": row["review_bucket"],
                "review_priority": row["review_priority"],
                "review_id": clip_id,
                "review_file": row["clip_path"],
                "source_audio": row["source_audio"],
                "source_start_time": float(row["start_time"]),
                "source_end_time": float(row["end_time"]),
                "derived_review_class": row["derived_review_class"],
                "v3_cough_prob": row["v3_cough_prob"],
                "v3_decision_075": row["v3_decision_075"],
                "v3_decision_095": row["v3_decision_095"],
                "yamnet_top1": row["yamnet_top1"],
                "yamnet_top5": row["yamnet_top5"],
                "yamnet_cough_score": row["yamnet_cough_score"],
                "opera_cough_score": "",
                "opera_hard_negative_score": "",
                "opera_rank": "",
                "manual_notes": row["manual_notes"],
            }
        )
        new_rows.append(values)

    added = pd.DataFrame(new_rows, columns=columns)
    return pd.concat([base_labels[columns], added], ignore_index=True)


def build_sessions(labels: pd.DataFrame) -> pd.DataFrame:
    columns = ["session_id", "person_id", "source_domain", "scenario_hint", "source"]
    for column in columns:
        if column not in labels.columns:
            labels[column] = ""
    sessions = labels[columns].drop_duplicates("session_id").copy()
    return sessions.sort_values("session_id").reset_index(drop=True)


def counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def bool_value(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def require_addon_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in ADDON_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required addon columns: {missing}")


def label_from_manual(decision: str) -> tuple[str, str]:
    if decision == "cough":
        return "cough", ""
    if decision == "hard_negative":
        return "non_cough", "hard_negative"
    if decision == "clean_non_cough":
        return "non_cough", "clean_non_cough"
    raise ValueError(f"Cannot map manual_decision to candidate label: {decision}")


def addon_crosstab(frame: pd.DataFrame, row: str, column: str) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame[row], frame[column], dropna=False)
    return {str(index): {str(key): int(value) for key, value in values.items()} for index, values in table.iterrows()}


def write_addon_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage v3.1-CANDIDATE-1 Summary",
        "",
        "This stage builds a board_live_data_1 candidate label addon from human review QA results only. "
        "It does not train, modify raw data, overwrite `audio_labels_v3.csv`, or produce a final merged label set.",
        "",
        "## Outputs",
        "",
        f"- candidate_csv: `{report['candidate_csv']}`",
        f"- report_json: `{report['report_json']}`",
        "",
        "## Counts",
        "",
        f"- total_input_rows: `{report['total_input_rows']}`",
        f"- candidate_rows: `{report['candidate_rows']}`",
        f"- excluded_rows: `{report['excluded_rows']}`",
        "",
        "## Label Counts",
        "",
    ]
    for key, value in report["label_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Negative Type Counts", ""])
    for key, value in report["negative_type_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Manual Decision Counts", ""])
    for key, value in report["manual_decision_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Person Distribution", ""])
    for key, value in report["person_id_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Use Constraint",
            "",
            "- All rows are `person_id=p01`; this addon can be used as board-live train addon or regression check, not as a standalone generalization evaluation set.",
            "- Canonical training windows are the original 1.0s source windows from `original_path/source_audio + window_start_sec/window_end_sec`; review clips are listening aids only.",
            "",
            "## Not Executed",
            "",
        ]
    )
    for item in report["not_executed"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_addon_only(args: argparse.Namespace) -> int:
    review_sheets = args.review_sheet or []
    if len(review_sheets) != 1:
        raise ValueError("--addon-only requires exactly one --review-sheet")
    review_sheet_path = resolve_path(review_sheets[0])
    if not review_sheet_path.exists():
        raise FileNotFoundError(f"Review sheet not found: {review_sheet_path}")

    out_labels_path = resolve_path(
        args.out_labels
        or (review_sheet_path.parent / "audio_labels_v3_1_candidate_board_live_data_1.csv")
    )
    report_json_path = resolve_path(args.out_report_json or (review_sheet_path.parent / "candidate_build_report.json"))
    summary_path = resolve_path(args.summary_md or (review_sheet_path.parent / "candidate_build_summary.md"))
    for path in (out_labels_path, report_json_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")

    frame, encoding = read_csv_with_fallback(review_sheet_path)
    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed")]
    require_addon_columns(frame, review_sheet_path)
    frame["manual_decision_norm"] = frame["manual_decision"].map(normalize_manual)
    invalid = sorted(set(frame["manual_decision_norm"]) - VALID_MANUAL_DECISIONS)
    if invalid:
        raise ValueError(f"{review_sheet_path} has invalid manual_decision values: {invalid}")
    blank_decisions = int((frame["manual_decision_norm"] == "").sum())
    if blank_decisions:
        raise ValueError(f"{review_sheet_path} has blank manual_decision rows: {blank_decisions}")

    frame["candidate_eligible_bool"] = frame["candidate_eligible"].map(bool_value)
    frame["candidate_exclude_bool"] = frame["candidate_exclude"].map(bool_value)
    eligible = frame[
        frame["candidate_eligible_bool"]
        & (~frame["candidate_exclude_bool"])
        & frame["manual_decision_norm"].isin(ADDON_ELIGIBLE_DECISIONS)
    ].copy()
    excluded = frame[
        frame["candidate_exclude_bool"] | frame["manual_decision_norm"].isin(ADDON_EXCLUDE_DECISIONS)
    ].copy()
    inconsistent = frame[
        frame["candidate_eligible_bool"] & frame["candidate_exclude_bool"]
    ]
    if not inconsistent.empty:
        raise ValueError(f"Rows cannot be both candidate_eligible and candidate_exclude: {len(inconsistent)}")
    if len(eligible) + len(excluded) != len(frame):
        raise ValueError(
            f"Candidate eligibility/exclusion does not partition input rows: "
            f"eligible={len(eligible)} excluded={len(excluded)} total={len(frame)}"
        )

    rows: list[dict[str, object]] = []
    for _, row in eligible.iterrows():
        decision = str(row["manual_decision_norm"])
        label, negative_type = label_from_manual(decision)
        original_path = str(row["original_path"]).strip()
        source_audio = str(row["source_audio"]).strip()
        canonical_audio_path = original_path or source_audio
        window_start = str(row["window_start_sec"]).strip()
        window_end = str(row["window_end_sec"]).strip()
        if not canonical_audio_path or not source_audio or not window_start or not window_end:
            raise ValueError("original_path/source_audio/window_start_sec/window_end_sec must be non-empty")
        candidate_id = stable_id(
            "board_live_data_1",
            canonical_audio_path,
            window_start,
            window_end,
            decision,
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "clip_id": row.get("clip_id", candidate_id),
                "label": label,
                "negative_type": negative_type,
                "source_domain": row["source_domain"],
                "person_id": row["person_id"],
                "session_id": row["session_id"],
                "original_path": original_path,
                "source_audio": source_audio,
                "canonical_audio_path": canonical_audio_path,
                "window_start_sec": float(row["window_start_sec"]),
                "window_end_sec": float(row["window_end_sec"]),
                "clip_path": row["clip_path"],
                "review_bucket": row["review_bucket"],
                "manual_decision": decision,
                "manual_subtype_raw": row["manual_subtype_raw"],
                "manual_subtype_norm": row["manual_subtype_norm"],
                "v3_cough_prob": row["v3_cough_prob"],
                "yamnet_top1": row["yamnet_top1"],
                "yamnet_cough_score": row["yamnet_cough_score"],
                "yamnet_speech_score": row["yamnet_speech_score"],
                "yamnet_throat_like_score": row["yamnet_throat_like_score"],
                "candidate_eligible": True,
                "candidate_exclude": False,
                "review_clip_is_training_source": False,
                "canonical_training_window": "original_1s_window",
            }
        )

    candidates = pd.DataFrame(rows, columns=ADDON_OUTPUT_COLUMNS)
    ensure_dir(out_labels_path.parent)
    candidates.to_csv(out_labels_path, index=False, encoding="utf-8-sig")

    input_person_counts = counts(frame["person_id"])
    candidate_person_counts = counts(candidates["person_id"]) if not candidates.empty else {}
    all_input_p01 = set(frame["person_id"].astype(str)) == {"p01"}
    all_candidate_p01 = set(candidates["person_id"].astype(str)) == {"p01"} if not candidates.empty else False
    report = {
        "stage": "v3.1-CANDIDATE-1",
        "input_review_sheet": str(review_sheet_path),
        "review_sheet_encoding": encoding,
        "candidate_csv": str(out_labels_path),
        "report_json": str(report_json_path),
        "summary_md": str(summary_path),
        "total_input_rows": int(len(frame)),
        "candidate_rows": int(len(candidates)),
        "excluded_rows": int(len(excluded)),
        "excluded_manual_decision_counts": counts(excluded["manual_decision_norm"]) if not excluded.empty else {},
        "label_counts": counts(candidates["label"]) if not candidates.empty else {},
        "manual_decision_counts": counts(frame["manual_decision_norm"]),
        "candidate_manual_decision_counts": counts(candidates["manual_decision"]) if not candidates.empty else {},
        "negative_type_counts": counts(candidates[candidates["negative_type"] != ""]["negative_type"]) if not candidates.empty else {},
        "manual_subtype_norm_counts": counts(candidates["manual_subtype_norm"]) if not candidates.empty else {},
        "review_bucket_by_label": addon_crosstab(candidates, "review_bucket", "label") if not candidates.empty else {},
        "person_id_counts": input_person_counts,
        "candidate_person_id_counts": candidate_person_counts,
        "all_input_person_id_p01": all_input_p01,
        "all_candidate_person_id_p01": all_candidate_p01,
        "person_id_warning": "All input and candidate rows are person_id=p01; do not use this addon alone as a generalization evaluation set.",
        "use_constraint": "Use only as a board_live train addon or regression check; not a standalone generalization evaluation set.",
        "canonical_training_window": "Use original_path/source_audio with window_start_sec/window_end_sec. Review clips are human-listening aids only.",
        "not_executed": ADDON_NOT_EXECUTED,
    }
    ensure_dir(report_json_path.parent)
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ensure_dir(summary_path.parent)
    write_addon_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# audio_labels_v3_1_candidate from board/YAMNet manual review",
        "",
        "This is a candidate label build only. It does not overwrite `audio_labels_v3.csv` or train a model.",
        "",
        "## Summary",
        "",
        f"- base_labels: `{report['base_labels']}`",
        f"- out_labels: `{report['out_labels']}`",
        f"- out_sessions: `{report['out_sessions']}`",
        f"- out_merged: `{report['out_merged']}`",
        f"- base_rows: `{report['base_rows']}`",
        f"- reviewed_rows: `{report['reviewed_rows']}`",
        f"- added_rows: `{report['added_rows']}`",
        f"- output_rows: `{report['output_rows']}`",
        f"- skipped_rows: `{report['skipped_rows']}`",
        f"- manual_conflict_rows: `{report['manual_conflict_rows']}`",
        f"- duplicate_skipped_rows: `{report['duplicate_skipped_rows']}`",
        "",
        "## Added Label Counts",
        "",
    ]
    for key, value in report["added_label_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Added hard_negative_source Counts", ""])
    for key, value in report["added_hard_negative_source_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Derived Review Class Counts", ""])
    for key, value in report["derived_review_class_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Manual Decision Counts", ""])
    for key, value in report["manual_decision_counts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Skipped Rows By Derived Class", ""])
    if report["skipped_derived_review_class_counts"]:
        for key, value in report["skipped_derived_review_class_counts"].items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.addon_only:
        return run_addon_only(args)

    config = load_config(args.config)
    paths = config["paths"]
    base_labels_path = resolve_path(args.base_labels)
    review_sheets = args.review_sheet or [str(resolve_path(paths["review_dir"]) / "review_sheet.csv")]
    out_labels_path = resolve_path(args.out_labels or paths["candidate_labels_csv"])
    out_sessions_path = resolve_path(args.out_sessions or paths["candidate_sessions_csv"])
    out_report_path = resolve_path(args.out_report or paths["candidate_report_md"])
    out_merged_path = resolve_path(args.out_merged or (Path(paths["review_dir"]) / "manual_review_merged.csv"))
    out_report_json = out_report_path.with_suffix(".json")

    if not base_labels_path.exists():
        raise FileNotFoundError(f"Base labels not found: {base_labels_path}")
    if out_labels_path.resolve() == base_labels_path.resolve():
        raise ValueError("--out-labels must not overwrite --base-labels")
    for path in (out_labels_path, out_sessions_path, out_merged_path, out_report_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")

    base_labels = pd.read_csv(base_labels_path, keep_default_na=False)
    merged, encodings = load_review_sheets(review_sheets)
    ensure_dir(out_merged_path.parent)
    merged.drop(columns=["_first_duplicate"], errors="ignore").to_csv(out_merged_path, index=False, encoding="utf-8-sig")

    candidates = merged[merged["include_in_v3_1_candidate"]].copy()
    output_labels = build_label_rows(base_labels.copy(), candidates)
    sessions = build_sessions(output_labels.copy())
    ensure_dir(out_labels_path.parent)
    output_labels.to_csv(out_labels_path, index=False, encoding="utf-8")
    ensure_dir(out_sessions_path.parent)
    sessions.to_csv(out_sessions_path, index=False, encoding="utf-8")

    reviewed = merged[merged["manual_decision_norm"] != ""]
    skipped = merged[(merged["manual_decision_norm"] != "") & (~merged["include_in_v3_1_candidate"])]
    report = {
        "base_labels": str(base_labels_path),
        "review_sheets": [str(resolve_path(path)) for path in review_sheets],
        "review_sheet_encodings": encodings,
        "out_labels": str(out_labels_path),
        "out_sessions": str(out_sessions_path),
        "out_merged": str(out_merged_path),
        "out_report": str(out_report_path),
        "base_rows": int(len(base_labels)),
        "merged_rows": int(len(merged)),
        "reviewed_rows": int(len(reviewed)),
        "unreviewed_rows": int(len(merged) - len(reviewed)),
        "added_rows": int(len(candidates)),
        "output_rows": int(len(output_labels)),
        "sessions_rows": int(len(sessions)),
        "skipped_rows": int(len(skipped)),
        "manual_conflict_rows": int(merged["manual_conflict"].sum()),
        "duplicate_skipped_rows": int(((~merged["_first_duplicate"]) & (merged["manual_decision_norm"] != "")).sum()),
        "manual_decision_counts": counts(merged["manual_decision_norm"].replace("", "<blank>")),
        "derived_review_class_counts": counts(merged["derived_review_class"]),
        "added_label_counts": counts(candidates["label"]) if not candidates.empty else {},
        "added_hard_negative_source_counts": counts(candidates["hard_negative_source"]) if not candidates.empty else {},
        "skipped_derived_review_class_counts": counts(skipped["derived_review_class"]) if not skipped.empty else {},
    }
    ensure_dir(out_report_path.parent)
    write_markdown(out_report_path, report)
    out_report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
