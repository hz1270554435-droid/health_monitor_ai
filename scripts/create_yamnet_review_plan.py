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


PLAN_COLUMNS = [
    "plan_bucket",
    "plan_rank",
    "rank_score",
    "source_selected_count",
    "manual_decision",
    "notes",
]

REQUIRED_COLUMNS = [
    "sample_id",
    "audio_file",
    "start_time",
    "end_time",
    "duration",
    "yamnet_top1",
    "yamnet_top1_score",
    "yamnet_cough_score",
    "yamnet_sneeze_score",
    "yamnet_speech_score",
    "yamnet_laughter_score",
    "yamnet_breathing_score",
    "yamnet_throat_like_score",
    "baseline_cough_prob",
    "decision",
    "review_priority",
    "sample_weight",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-limited CSV review plan from YAMNet candidate_pool.csv."
    )
    parser.add_argument(
        "--candidate-csv",
        default="data/mining_candidates/yamnet_non_cough_full_resumable/candidate_pool.csv",
    )
    parser.add_argument("--out-dir", default="review/yamnet_full_non_cough_v1")
    parser.add_argument("--top-hard", type=int, default=1000)
    parser.add_argument("--top-baseline-suspicious", type=int, default=1500)
    parser.add_argument("--top-uncertain", type=int, default=500)
    parser.add_argument("--top-pseudo-non-cough", type=int, default=500)
    parser.add_argument(
        "--top-pseudo-cough",
        type=int,
        default=-1,
        help="Use -1 to include all pseudo_cough_candidate rows after source limits.",
    )
    parser.add_argument("--max-per-source", type=int, default=5)
    parser.add_argument("--min-start-gap-sec", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-csv",
        action="append",
        default=[],
        help=(
            "CSV with previously selected/reviewed rows to exclude. Can be passed multiple times. "
            "Supports either audio_file/start_time/end_time or source_audio/source_start_time/source_end_time."
        ),
    )
    parser.add_argument(
        "--exclude-mode",
        choices=["all", "reviewed"],
        default="all",
        help="When the exclude CSV has manual_decision, exclude all rows or only rows with a non-empty manual_decision.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def stable_random_score(value: object, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(0xFFFFFFFFFFFF)


def validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def row_key(audio_file: object, start_time: object, end_time: object) -> tuple[str, float, float]:
    return (str(audio_file), round(float(start_time), 6), round(float(end_time), 6))


def candidate_keys(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: row_key(row["audio_file"], row["start_time"], row["end_time"]), axis=1)


def load_exclude_keys(paths: list[str], mode: str) -> tuple[set[tuple[str, float, float]], dict[str, int]]:
    keys: set[tuple[str, float, float]] = set()
    stats: dict[str, int] = {}
    for raw_path in paths:
        path = resolve_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Exclude CSV not found: {path}")
        df = pd.read_csv(path, keep_default_na=False)
        if mode == "reviewed" and "manual_decision" in df.columns:
            df = df[df["manual_decision"].astype(str).str.strip() != ""].copy()

        if {"audio_file", "start_time", "end_time"}.issubset(df.columns):
            audio_col, start_col, end_col = "audio_file", "start_time", "end_time"
        elif {"source_audio", "source_start_time", "source_end_time"}.issubset(df.columns):
            audio_col, start_col, end_col = "source_audio", "source_start_time", "source_end_time"
        else:
            raise ValueError(
                f"{path} must contain either audio_file/start_time/end_time "
                "or source_audio/source_start_time/source_end_time"
            )

        before = len(keys)
        for _, row in df.iterrows():
            keys.add(row_key(row[audio_col], row[start_col], row[end_col]))
        stats[str(path)] = len(keys) - before
    return keys, stats


def sort_candidates(df: pd.DataFrame, decision: str, threshold: float, seed: int) -> pd.DataFrame:
    subset = df[df["decision"] == decision].copy()
    if subset.empty:
        subset["rank_score"] = []
        return subset

    if decision in {"hard_negative_review", "baseline_suspicious_review"}:
        subset["rank_score"] = subset["baseline_cough_prob"].astype(float)
        return subset.sort_values(["rank_score", "yamnet_top1_score"], ascending=[False, False])
    if decision == "pseudo_cough_candidate":
        subset["rank_score"] = subset["baseline_cough_prob"].astype(float) + subset["yamnet_cough_score"].astype(float)
        return subset.sort_values(["rank_score", "baseline_cough_prob"], ascending=[False, False])
    if decision == "uncertain_review":
        subset["rank_score"] = -(subset["baseline_cough_prob"].astype(float) - float(threshold)).abs()
        return subset.sort_values(["rank_score", "baseline_cough_prob"], ascending=[False, False])
    if decision == "pseudo_non_cough_candidate":
        subset["rank_score"] = subset["sample_id"].map(lambda value: stable_random_score(value, seed))
        return subset.sort_values("rank_score", ascending=False)

    subset["rank_score"] = 0.0
    return subset


def can_take(
    row: pd.Series,
    source_counts: dict[str, int],
    source_times: dict[str, list[float]],
    max_per_source: int,
    min_start_gap_sec: float,
) -> bool:
    source = str(row["audio_file"])
    if source_counts.get(source, 0) >= max_per_source:
        return False
    start_time = float(row["start_time"])
    for selected_time in source_times.get(source, []):
        if abs(start_time - selected_time) < min_start_gap_sec:
            return False
    return True


def select_bucket(
    sorted_rows: pd.DataFrame,
    bucket_name: str,
    quota: int,
    source_counts: dict[str, int],
    source_times: dict[str, list[float]],
    max_per_source: int,
    min_start_gap_sec: float,
) -> pd.DataFrame:
    if quota == 0 or sorted_rows.empty:
        return sorted_rows.head(0).copy()

    selected_rows: list[pd.Series] = []
    limit = len(sorted_rows) if quota < 0 else quota
    for _, row in sorted_rows.iterrows():
        if len(selected_rows) >= limit:
            break
        if not can_take(row, source_counts, source_times, max_per_source, min_start_gap_sec):
            continue
        source = str(row["audio_file"])
        source_counts[source] = source_counts.get(source, 0) + 1
        source_times.setdefault(source, []).append(float(row["start_time"]))
        output = row.copy()
        output["plan_bucket"] = bucket_name
        output["plan_rank"] = len(selected_rows) + 1
        output["source_selected_count"] = source_counts[source]
        output["manual_decision"] = ""
        output["notes"] = ""
        selected_rows.append(output)

    if not selected_rows:
        return sorted_rows.head(0).copy()
    return pd.DataFrame(selected_rows)


def write_report(
    report_path: Path,
    candidate_csv: Path,
    out_csv: Path,
    selected: pd.DataFrame,
    source_counts: dict[str, int],
    candidate_counts: dict[str, int],
    args: argparse.Namespace,
) -> None:
    ensure_dir(report_path.parent)
    lines = [
        "# YAMNet full non-cough review plan v1",
        "",
        "This plan is CSV-only. It does not export audio, modify labels, or train a model.",
        "",
        f"- candidate_csv: `{candidate_csv}`",
        f"- review_plan_csv: `{out_csv}`",
        f"- selected_rows: {len(selected)}",
        f"- selected_sources: {selected['audio_file'].nunique() if not selected.empty else 0}",
        f"- max_per_source: {args.max_per_source}",
        f"- min_start_gap_sec: {args.min_start_gap_sec}",
        f"- threshold: {args.threshold}",
        f"- exclude_mode: {args.exclude_mode}",
        f"- exclude_csv: {args.exclude_csv}",
        "",
        "## Candidate decision counts",
        "",
    ]
    for decision, count in candidate_counts.items():
        lines.append(f"- {decision}: {count}")

    lines.extend(["", "## Review plan buckets", ""])
    if selected.empty:
        lines.append("_None._")
    else:
        for bucket, count in selected["plan_bucket"].value_counts().sort_index().items():
            lines.append(f"- {bucket}: {int(count)}")

    lines.extend(["", "## Top YAMNet top1 labels in selected rows", ""])
    if selected.empty:
        lines.append("_None._")
    else:
        for label, count in selected["yamnet_top1"].value_counts().head(30).items():
            lines.append(f"- {label}: {int(count)}")

    lines.extend(["", "## Source concentration", ""])
    if not source_counts:
        lines.append("_None._")
    else:
        counts = pd.Series(source_counts)
        lines.extend(
            [
                f"- max_selected_per_source: {int(counts.max())}",
                f"- sources_at_limit: {int((counts >= args.max_per_source).sum())}",
            ]
        )

    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Inspect this CSV before exporting WAV clips. If the distribution looks reasonable, export review clips from this plan.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_per_source <= 0:
        raise ValueError("--max-per-source must be positive")
    if args.min_start_gap_sec < 0:
        raise ValueError("--min-start-gap-sec must be non-negative")

    candidate_csv = resolve_path(args.candidate_csv)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    out_csv = out_dir / "review_plan.csv"
    report_md = out_dir / "review_plan_report.md"
    report_json = out_dir / "review_plan_report.json"

    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")

    df = pd.read_csv(candidate_csv, usecols=lambda column: column in REQUIRED_COLUMNS)
    validate_columns(df, candidate_csv)
    candidate_counts = {str(k): int(v) for k, v in df["decision"].value_counts().items()}
    exclude_keys, exclude_stats = load_exclude_keys(args.exclude_csv, args.exclude_mode)
    rows_before_exclude = len(df)
    if exclude_keys:
        keys = candidate_keys(df)
        df = df[~keys.isin(exclude_keys)].copy()
    rows_after_exclude = len(df)

    source_counts: dict[str, int] = {}
    source_times: dict[str, list[float]] = {}
    bucket_specs = [
        ("hard_negative_review", "hard_negative_review", args.top_hard),
        ("baseline_suspicious_review", "baseline_suspicious_review", args.top_baseline_suspicious),
        ("pseudo_cough_candidate", "pseudo_cough_candidate", args.top_pseudo_cough),
        ("uncertain_review", "uncertain_review", args.top_uncertain),
        ("pseudo_non_cough_candidate", "pseudo_non_cough_candidate", args.top_pseudo_non_cough),
    ]

    selected_parts: list[pd.DataFrame] = []
    for decision, bucket_name, quota in bucket_specs:
        sorted_rows = sort_candidates(df, decision, float(args.threshold), int(args.seed))
        selected_parts.append(
            select_bucket(
                sorted_rows,
                bucket_name=bucket_name,
                quota=int(quota),
                source_counts=source_counts,
                source_times=source_times,
                max_per_source=int(args.max_per_source),
                min_start_gap_sec=float(args.min_start_gap_sec),
            )
        )

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    output_columns = [*PLAN_COLUMNS, *REQUIRED_COLUMNS]
    if selected.empty:
        selected = pd.DataFrame(columns=output_columns)
    else:
        selected = selected[output_columns]
    selected.to_csv(out_csv, index=False)

    report = {
        "candidate_csv": str(candidate_csv),
        "review_plan_csv": str(out_csv),
        "review_plan_report_md": str(report_md),
        "selected_rows": int(len(selected)),
        "selected_sources": int(selected["audio_file"].nunique()) if not selected.empty else 0,
        "candidate_decision_counts": candidate_counts,
        "rows_before_exclude": int(rows_before_exclude),
        "rows_after_exclude": int(rows_after_exclude),
        "excluded_rows": int(rows_before_exclude - rows_after_exclude),
        "exclude_mode": str(args.exclude_mode),
        "exclude_csv_stats": exclude_stats,
        "review_plan_counts": {
            str(k): int(v) for k, v in selected["plan_bucket"].value_counts().items()
        }
        if not selected.empty
        else {},
        "max_per_source": int(args.max_per_source),
        "min_start_gap_sec": float(args.min_start_gap_sec),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(report_md, candidate_csv, out_csv, selected, source_counts, candidate_counts, args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
