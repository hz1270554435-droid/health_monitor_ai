from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import stable_hash


COUGH_LIKE_TOP1 = {"Cough", "Snort", "Breathing"}
ROOM_OR_SILENCE_TOP1 = {"Silence", "Inside, small room"}
MID_CONFUSION_TOP1 = {
    "Speech",
    "Cough",
    "Snort",
    "Breathing",
    "Silence",
    "Inside, small room",
}

BUCKET_ORDER = [
    "P0_hard_negative_speech",
    "P0_hard_negative_room_or_silence",
    "P1_possible_true_cough",
    "P1_possible_v3_false_negative",
    "P1_high_v3_other_yamnet",
    "P2_mid_confusion",
    "P3_random_background",
]

FULL_KEEP_BUCKETS = {
    "P0_hard_negative_speech",
    "P0_hard_negative_room_or_silence",
    "P1_possible_true_cough",
    "P1_possible_v3_false_negative",
    "P1_high_v3_other_yamnet",
}

REVIEW_PRIORITY = {
    "P0_hard_negative_speech": "P0",
    "P0_hard_negative_room_or_silence": "P0",
    "P1_possible_true_cough": "P1",
    "P1_possible_v3_false_negative": "P1",
    "P1_high_v3_other_yamnet": "P1",
    "P2_mid_confusion": "P2",
    "P3_random_background": "P3",
}

REQUIRED_COLUMNS = [
    "original_path",
    "audio_file",
    "session_id",
    "person_id",
    "window_start_sec",
    "window_end_sec",
    "yamnet_top1",
    "v3_cough_prob",
]

REQUIRED_NONEMPTY_COLUMNS = [
    "original_path",
    "session_id",
    "person_id",
    "window_start_sec",
    "window_end_sec",
]

NOT_EXECUTED = [
    "export_v3_1_review_clips.py",
    "build_v3_1_candidate_from_manual_reviews.py",
    "audio_labels_v3_1_candidate.csv generation",
    "training",
    "raw audio copy",
    "ml/data/raw modification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic v3.1 board-live manual review plan from v3/YAMNet scores."
    )
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--manifest", required=True, help="Input v3_scored_manifest.csv")
    parser.add_argument("--out", required=True, help="Output review_plan.csv")
    parser.add_argument("--report", default=None, help="Output review_plan_report.json")
    parser.add_argument("--summary-md", default=None, help="Output review_plan_summary.md")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--p2-quota", type=int, default=120)
    parser.add_argument("--p3-quota", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def to_float(value: object, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_random_score(seed: int, bucket: str, key: str) -> float:
    digest = stable_hash(seed, bucket, key, length=12)
    return int(digest, 16) / float(0xFFFFFFFFFFFF)


def row_key(row: pd.Series) -> str:
    return (
        f"{row.get('original_path', '')}|"
        f"{to_float(row.get('window_start_sec'), 0.0):.6f}|"
        f"{to_float(row.get('window_end_sec'), 0.0):.6f}"
    )


def validate_input(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Input manifest is missing required columns: {missing}")

    empty_required = {
        column: int((frame[column].astype(str).str.strip() == "").sum()) for column in REQUIRED_NONEMPTY_COLUMNS
    }
    bad_empty = {key: value for key, value in empty_required.items() if value > 0}
    if bad_empty:
        raise ValueError(f"Input manifest has empty required fields: {bad_empty}")

    manual_present = "manual_decision" in frame.columns
    manual_all_empty = True
    nonempty_manual = 0
    if manual_present:
        manual_values = frame["manual_decision"].astype(str).str.strip()
        nonempty_manual = int((manual_values != "").sum())
        manual_all_empty = nonempty_manual == 0
        if not manual_all_empty:
            raise ValueError(f"manual_decision exists but is not empty for {nonempty_manual} rows")

    keys = frame.apply(row_key, axis=1)
    unique_key_count = int(keys.nunique())
    duplicate_key_count = int(len(keys) - unique_key_count)
    if duplicate_key_count:
        raise ValueError(
            "Input manifest has duplicate window keys "
            f"(original_path + window_start_sec + window_end_sec): {duplicate_key_count}"
        )

    return {
        "row_count": int(len(frame)),
        "unique_key_count": unique_key_count,
        "duplicate_key_count": duplicate_key_count,
        "manual_decision_present": manual_present,
        "manual_decision_all_empty": manual_all_empty,
        "manual_decision_nonempty_count": nonempty_manual,
        "required_nonempty_empty_counts": empty_required,
    }


def classify_bucket(row: pd.Series) -> str:
    v3_prob = to_float(row.get("v3_cough_prob"))
    yamnet_top1 = str(row.get("yamnet_top1", ""))

    if v3_prob >= 0.90 and yamnet_top1 == "Speech":
        return "P0_hard_negative_speech"
    if v3_prob >= 0.90 and yamnet_top1 in ROOM_OR_SILENCE_TOP1:
        return "P0_hard_negative_room_or_silence"
    if v3_prob >= 0.90 and yamnet_top1 in COUGH_LIKE_TOP1:
        return "P1_possible_true_cough"
    if v3_prob < 0.50 and yamnet_top1 in COUGH_LIKE_TOP1:
        return "P1_possible_v3_false_negative"
    if v3_prob >= 0.90:
        return "P1_high_v3_other_yamnet"
    if 0.50 <= v3_prob < 0.90 and yamnet_top1 in MID_CONFUSION_TOP1:
        return "P2_mid_confusion"
    if v3_prob < 0.50:
        return "P3_random_background"
    return ""


def add_review_fields(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["review_bucket"] = output.apply(classify_bucket, axis=1)
    output["review_priority"] = output["review_bucket"].map(REVIEW_PRIORITY).fillna("")
    return output


def sort_full_keep(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    output["_v3_sort"] = output["v3_cough_prob"].astype(float)
    output["_window_sort"] = output["window_start_sec"].astype(float)
    output = output.sort_values(
        ["review_bucket", "_v3_sort", "session_id", "_window_sort"],
        ascending=[True, False, True, True],
    )
    return output.drop(columns=["_v3_sort", "_window_sort"])


def select_round_robin(frame: pd.DataFrame, bucket: str, quota: int, seed: int) -> pd.DataFrame:
    subset = frame[frame["review_bucket"] == bucket].copy()
    if subset.empty or quota <= 0:
        return subset.head(0)

    subset["_key"] = subset.apply(row_key, axis=1)
    subset["_rand"] = subset["_key"].map(lambda key: stable_random_score(seed, bucket, key))
    subset["_v3_sort"] = subset["v3_cough_prob"].astype(float)
    subset["_window_sort"] = subset["window_start_sec"].astype(float)

    queues: list[deque[pd.Series]] = []
    for _, group in subset.sort_values(["session_id", "_rand", "_v3_sort"]).groupby("session_id", sort=True):
        ordered = group.sort_values(["_rand", "_v3_sort", "_window_sort"], ascending=[True, False, True])
        queues.append(deque(row for _, row in ordered.iterrows()))

    selected: list[pd.Series] = []
    while queues and len(selected) < quota:
        next_queues: list[deque[pd.Series]] = []
        for queue in queues:
            if not queue or len(selected) >= quota:
                continue
            selected.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues

    if not selected:
        return subset.head(0).drop(columns=["_key", "_rand", "_v3_sort", "_window_sort"], errors="ignore")
    return pd.DataFrame(selected).drop(columns=["_key", "_rand", "_v3_sort", "_window_sort"], errors="ignore")


def select_review_plan(candidates: pd.DataFrame, p2_quota: int, p3_quota: int, seed: int) -> pd.DataFrame:
    selected_parts: list[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        bucket_rows = candidates[candidates["review_bucket"] == bucket]
        if bucket in FULL_KEEP_BUCKETS:
            selected_parts.append(sort_full_keep(bucket_rows))
        elif bucket == "P2_mid_confusion":
            selected_parts.append(select_round_robin(candidates, bucket, p2_quota, seed))
        elif bucket == "P3_random_background":
            selected_parts.append(select_round_robin(candidates, bucket, p3_quota, seed))

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else candidates.head(0)
    selected["_bucket_order"] = selected["review_bucket"].map({bucket: idx for idx, bucket in enumerate(BUCKET_ORDER)})
    selected["_v3_sort"] = selected["v3_cough_prob"].astype(float)
    selected["_window_sort"] = selected["window_start_sec"].astype(float)
    selected = selected.sort_values(
        ["_bucket_order", "session_id", "_v3_sort", "_window_sort"],
        ascending=[True, True, False, True],
    ).drop(columns=["_bucket_order", "_v3_sort", "_window_sort"])
    return selected.reset_index(drop=True)


def series_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def score_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    values = frame["v3_cough_prob"].astype(float)
    return {
        "count": int(values.count()),
        "min": round(float(values.min()), 9),
        "mean": round(float(values.mean()), 9),
        "p50": round(float(values.quantile(0.50)), 9),
        "p90": round(float(values.quantile(0.90)), 9),
        "p95": round(float(values.quantile(0.95)), 9),
        "p99": round(float(values.quantile(0.99)), 9),
        "max": round(float(values.max()), 9),
    }


def top_rows(frame: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    columns = [
        "review_bucket",
        "review_priority",
        "session_id",
        "window_id",
        "window_start_sec",
        "window_end_sec",
        "yamnet_top1",
        "yamnet_top5",
        "v3_cough_prob",
        "v3_non_cough_prob",
        "original_path",
    ]
    available = [column for column in columns if column in frame.columns]
    rows = frame.sort_values("v3_cough_prob", ascending=False).head(limit)
    return rows[available].to_dict(orient="records")


def nested_bucket_session_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for bucket in BUCKET_ORDER:
        subset = frame[frame["review_bucket"] == bucket]
        if subset.empty:
            output[bucket] = {}
        else:
            output[bucket] = series_counts(subset["session_id"])
    return output


def bucket_score_summaries(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    return {bucket: score_summary(frame[frame["review_bucket"] == bucket]) for bucket in BUCKET_ORDER}


def bucket_yamnet_distributions(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for bucket in BUCKET_ORDER:
        subset = frame[frame["review_bucket"] == bucket]
        output[bucket] = series_counts(subset["yamnet_top1"]) if not subset.empty else {}
    return output


def build_report(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    summary_path: Path,
    seed: int,
    p2_quota: int,
    p3_quota: int,
    validation: dict[str, Any],
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "stage": "v3.1-REVIEWPLAN-1",
        "input_csv": str(input_path),
        "review_plan_csv": str(output_path),
        "report_json": str(report_path),
        "summary_md": str(summary_path),
        "random_seed": int(seed),
        "quota_settings": {
            "full_keep_buckets": sorted(FULL_KEEP_BUCKETS),
            "p2_quota": int(p2_quota),
            "p3_quota": int(p3_quota),
        },
        "total_input_rows": int(validation["row_count"]),
        "candidate_rows": int(len(candidates)),
        "unbucketed_rows": int(validation["row_count"] - len(candidates)),
        "total_selected_rows": int(len(selected)),
        "candidate_rows_per_bucket": series_counts(candidates["review_bucket"]),
        "selected_rows_per_bucket": series_counts(selected["review_bucket"]),
        "selected_rows_per_session": series_counts(selected["session_id"]),
        "selected_rows_per_bucket_session": nested_bucket_session_counts(selected),
        "v3_cough_prob_summary_per_bucket": bucket_score_summaries(selected),
        "yamnet_top1_distribution_per_bucket": bucket_yamnet_distributions(selected),
        "top_30_highest_v3_cough_prob_rows": top_rows(selected, limit=30),
        "manual_decision_present": bool(validation["manual_decision_present"]),
        "manual_decision_all_empty": bool(validation["manual_decision_all_empty"]),
        "manual_decision_nonempty_count": int(validation["manual_decision_nonempty_count"]),
        "input_validation": validation,
        "not_executed": NOT_EXECUTED,
    }


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage v3.1-REVIEWPLAN-1 Summary",
        "",
        "This review plan is generated from v3/YAMNet weak scores only. It does not export clips, "
        "create candidate labels, train a model, copy raw audio, or modify ml/data/raw/.",
        "",
        "## Outputs",
        "",
        f"- input_csv: `{report['input_csv']}`",
        f"- review_plan_csv: `{report['review_plan_csv']}`",
        f"- report_json: `{report['report_json']}`",
        "",
        "## Counts",
        "",
        f"- total_input_rows: `{report['total_input_rows']}`",
        f"- total_selected_rows: `{report['total_selected_rows']}`",
        f"- random_seed: `{report['random_seed']}`",
        "",
        "## Selected Rows Per Bucket",
        "",
    ]
    for bucket in BUCKET_ORDER:
        value = report["selected_rows_per_bucket"].get(bucket, 0)
        lines.append(f"- {bucket}: `{value}`")

    lines.extend(["", "## Selected Rows Per Session", ""])
    for session_id, value in report["selected_rows_per_session"].items():
        lines.append(f"- {session_id}: `{value}`")

    lines.extend(["", "## v3_cough_prob Summary Per Bucket", ""])
    for bucket, summary in report["v3_cough_prob_summary_per_bucket"].items():
        lines.append(
            "- "
            f"{bucket}: count=`{summary['count']}`, min=`{summary['min']}`, mean=`{summary['mean']}`, "
            f"p50=`{summary['p50']}`, p90=`{summary['p90']}`, p95=`{summary['p95']}`, "
            f"p99=`{summary['p99']}`, max=`{summary['max']}`"
        )

    lines.extend(["", "## YAMNet top1 Distribution Per Bucket", ""])
    for bucket, distribution in report["yamnet_top1_distribution_per_bucket"].items():
        rendered = ", ".join(f"{key}: {value}" for key, value in distribution.items()) or "none"
        lines.append(f"- {bucket}: {rendered}")

    lines.extend(["", "## Top 30 Highest v3_cough_prob Rows", ""])
    for row in report["top_30_highest_v3_cough_prob_rows"]:
        lines.append(
            "- "
            f"{row.get('review_bucket', '')} | {row.get('session_id', '')} | "
            f"{row.get('window_start_sec', '')}-{row.get('window_end_sec', '')}s | "
            f"yamnet_top1={row.get('yamnet_top1', '')} | "
            f"v3_cough_prob={row.get('v3_cough_prob', '')}"
        )

    lines.extend(
        [
            "",
            "## Manual Decision Policy",
            "",
            f"- manual_decision_present: `{report['manual_decision_present']}`",
            f"- manual_decision_all_empty: `{report['manual_decision_all_empty']}`",
            "- The script does not add manual_decision when the input does not already contain it.",
            "",
            "## Not Executed",
            "",
        ]
    )
    for item in report["not_executed"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def assert_outputs(selected: pd.DataFrame) -> None:
    if not (240 <= len(selected) <= 350):
        raise ValueError(f"Review plan row count is outside expected first-round range: {len(selected)}")
    if "review_bucket" not in selected.columns or int((selected["review_bucket"].astype(str).str.strip() == "").sum()):
        raise ValueError("review_bucket is missing or empty for selected rows")
    for column in REQUIRED_NONEMPTY_COLUMNS:
        if column not in selected.columns:
            raise ValueError(f"Selected review plan lost required column: {column}")
        empty_count = int((selected[column].astype(str).str.strip() == "").sum())
        if empty_count:
            raise ValueError(f"Selected review plan has empty {column}: {empty_count}")
    if "manual_decision" in selected.columns:
        nonempty = int((selected["manual_decision"].astype(str).str.strip() != "").sum())
        if nonempty:
            raise ValueError(f"manual_decision exists in output but is not empty for {nonempty} rows")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))

    manifest_path = resolve_path(args.manifest)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report) if args.report else out_path.with_name("review_plan_report.json")
    summary_path = resolve_path(args.summary_md) if args.summary_md else out_path.with_name("review_plan_summary.md")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Input v3 scored manifest not found: {manifest_path}")
    for path in (out_path, report_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")

    frame = pd.read_csv(manifest_path, keep_default_na=False)
    validation = validate_input(frame)
    candidates = add_review_fields(frame)
    candidates = candidates[candidates["review_bucket"] != ""].copy()
    selected = select_review_plan(candidates, p2_quota=args.p2_quota, p3_quota=args.p3_quota, seed=seed)
    assert_outputs(selected)

    ensure_dir(out_path.parent)
    selected.to_csv(out_path, index=False)

    report = build_report(
        input_path=manifest_path,
        output_path=out_path,
        report_path=report_path,
        summary_path=summary_path,
        seed=seed,
        p2_quota=args.p2_quota,
        p3_quota=args.p3_quota,
        validation=validation,
        candidates=candidates,
        selected=selected,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
