from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
THRESHOLD_COLS = [f"trigger_ge_{th:.2f}".replace(".", "_") for th in THRESHOLDS]
NOT_EXECUTED = [
    "training",
    "model export",
    "board deployment",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
    "review clip export",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize top suspicious windows from FP-fix live smoke scoring."
    )
    parser.add_argument(
        "--scored",
        default="review/board_live_data_1/fpfix_live_smoke_scored.csv",
        help="Input scored live-smoke CSV.",
    )
    parser.add_argument(
        "--report",
        default="review/board_live_data_1/fpfix_live_smoke_report.json",
        help="Existing live-smoke report path, recorded for traceability only.",
    )
    parser.add_argument(
        "--out",
        default="review/board_live_data_1/fpfix_live_smoke_top_windows.csv",
        help="Output CSV containing non-cough top windows and real-cough triggers.",
    )
    parser.add_argument(
        "--summary-md",
        default="review/board_live_data_1/fpfix_live_smoke_review_summary.md",
        help="Output Markdown summary for manual listening.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cough-threshold", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def gate_event_count(flags: list[bool], required_hits: int, span: int) -> int:
    if not flags:
        return 0
    gate_flags: list[bool] = []
    for idx in range(len(flags)):
        start = max(0, idx - span + 1)
        gate_flags.append(sum(flags[start : idx + 1]) >= required_hits)

    events = 0
    in_event = False
    for active in gate_flags:
        if active and not in_event:
            events += 1
            in_event = True
        elif not active:
            in_event = False
    return events


def max_true_run(flags: list[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def threshold_flags(row: pd.Series) -> str:
    parts = []
    for threshold, column in zip(THRESHOLDS, THRESHOLD_COLS):
        parts.append(f">={threshold:.2f}:{bool(row[column])}")
    return ", ".join(parts)


def quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def scenario_gate_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scenario, session_key), group in df.sort_values("window_start_sec").groupby(
        ["scenario_hint", "session_key"], dropna=False
    ):
        flags_050 = bool_series(group["trigger_ge_0_50"]).tolist()
        rows.append(
            {
                "scenario_hint": str(scenario),
                "session_key": str(session_key),
                "session_type": str(group["session_type"].iloc[0]),
                "windows": int(len(group)),
                "trigger_ge_0_50": int(sum(flags_050)),
                "max_true_run_ge_0_50": int(max_true_run(flags_050)),
                "gate_2_of_3_events_ge_0_50": int(gate_event_count(flags_050, 2, 3)),
                "gate_3_of_5_events_ge_0_50": int(gate_event_count(flags_050, 3, 5)),
                "max_fpfix_cough_prob": float(group["fpfix_cough_prob"].max()),
                "p95_fpfix_cough_prob": quantile(group["fpfix_cough_prob"], 0.95),
            }
        )
    return rows


def choose_recommendation(gate_rows: list[dict[str, Any]]) -> tuple[str, str]:
    non_cough = [row for row in gate_rows if row["session_type"] == "non_cough"]
    cough = [row for row in gate_rows if row["scenario_hint"] == "real_cough"]

    non_cough_gate_events = sum(row["gate_2_of_3_events_ge_0_50"] for row in non_cough)
    non_cough_has_nonisolated = any(row["max_true_run_ge_0_50"] > 1 for row in non_cough)
    real_cough_gate_events = sum(row["gate_2_of_3_events_ge_0_50"] for row in cough)

    if non_cough_gate_events > 0 or non_cough_has_nonisolated:
        return (
            "mine_new_false_alarms",
            "At least one non-cough session has a non-isolated trigger pattern or a 2-of-3 gated event.",
        )
    if real_cough_gate_events >= 1:
        return (
            "collect_more_smoke",
            "The 0.50 + 2-of-3 gate is supported by this smoke set, but there is only one cough session, no annotations, and quiet_room is prior copied background.",
        )
    return (
        "collect_more_smoke",
        "Non-cough triggers are suppressed, but the cough session did not form a 2-of-3 gated event at threshold 0.50.",
    )


def format_window_line(row: pd.Series) -> str:
    return (
        f"| {int(row['review_rank'])} | `{row['session_id']}` | "
        f"{float(row['window_start_sec']):.2f}-{float(row['window_end_sec']):.2f} | "
        f"{float(row['fpfix_cough_prob']):.6f} | "
        f"`{threshold_flags(row)}` | `{row['source_audio']}` |"
    )


def write_summary(
    summary_path: Path,
    scored_path: Path,
    report_path: Path,
    out_path: Path,
    df: pd.DataFrame,
    top_windows: pd.DataFrame,
    gate_rows: list[dict[str, Any]],
    recommendation: str,
    recommendation_reason: str,
) -> None:
    lines: list[str] = []
    lines.append("# v3.1 FP-fix Live Smoke Top Suspicious Windows")
    lines.append("")
    lines.append(f"- scored_csv: `{scored_path}`")
    lines.append(f"- source_report: `{report_path}`")
    lines.append(f"- top_windows_csv: `{out_path}`")
    lines.append(f"- input_rows: `{len(df)}`")
    lines.append(f"- output_rows: `{len(top_windows)}`")
    lines.append(f"- recommendation: `{recommendation}`")
    lines.append(f"- recommendation_reason: {recommendation_reason}")
    lines.append("- formal_generalization_evaluation: `False`")
    lines.append("- review_clips_exported: `False`")
    lines.append("")

    lines.append("## Gate Check")
    lines.append("")
    lines.append("| scenario | session | type | >=0.50 singles | max run | 2-of-3 events | 3-of-5 events | max prob | p95 prob |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in gate_rows:
        lines.append(
            f"| `{row['scenario_hint']}` | `{row['session_key']}` | `{row['session_type']}` | "
            f"{row['trigger_ge_0_50']} | {row['max_true_run_ge_0_50']} | "
            f"{row['gate_2_of_3_events_ge_0_50']} | {row['gate_3_of_5_events_ge_0_50']} | "
            f"{row['max_fpfix_cough_prob']:.6f} | {row['p95_fpfix_cough_prob']:.6f} |"
        )
    lines.append("")

    non_cough_top = top_windows[top_windows["row_kind"] == "non_cough_top5"]
    lines.append("## Non-Cough Top 5 Windows")
    for scenario, group in non_cough_top.groupby("scenario_hint", sort=True):
        lines.append("")
        lines.append(f"### {scenario}")
        lines.append("| rank | session | window sec | fpfix_cough_prob | threshold flags | source_audio |")
        lines.append("|---:|---|---:|---:|---|---|")
        for _, row in group.sort_values("review_rank").iterrows():
            lines.append(format_window_line(row))
    lines.append("")

    cough_rows = top_windows[top_windows["row_kind"] == "real_cough_ge_0_50"]
    lines.append("## Real Cough Windows >= 0.50")
    lines.append("")
    if cough_rows.empty:
        lines.append("- No real_cough windows reached threshold 0.50.")
    else:
        lines.append("| rank | session | window sec | fpfix_cough_prob | threshold flags | source_audio |")
        lines.append("|---:|---|---:|---:|---|---|")
        for _, row in cough_rows.sort_values("window_start_sec").iterrows():
            lines.append(format_window_line(row))
    lines.append("")

    candidate_supported = (
        sum(row["gate_2_of_3_events_ge_0_50"] for row in gate_rows if row["session_type"] == "non_cough") == 0
        and sum(row["gate_2_of_3_events_ge_0_50"] for row in gate_rows if row["scenario_hint"] == "real_cough") >= 1
    )
    lines.append("## Candidate Gate")
    lines.append("")
    lines.append(f"- current_candidate_strategy: `threshold=0.50 + 2-of-3 gate`")
    lines.append(f"- supported_by_this_smoke: `{candidate_supported}`")
    lines.append("- Caveat: this remains a fresh smoke replay check, not a formal generalization evaluation.")
    lines.append("")

    lines.append("## Not Executed")
    lines.append("")
    for item in NOT_EXECUTED:
        lines.append(f"- {item}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scored_path = resolve_path(args.scored)
    report_path = resolve_path(args.report)
    out_path = resolve_path(args.out)
    summary_path = resolve_path(args.summary_md)

    if not scored_path.exists():
        raise FileNotFoundError(f"Scored CSV not found: {scored_path}")
    for path in [out_path, summary_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {path}")

    df = pd.read_csv(scored_path)
    required = [
        "source_audio",
        "session_key",
        "session_id",
        "scenario_hint",
        "session_type",
        "window_start_sec",
        "window_end_sec",
        "fpfix_cough_prob",
        *THRESHOLD_COLS,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["fpfix_cough_prob"] = pd.to_numeric(df["fpfix_cough_prob"], errors="raise")
    if df["fpfix_cough_prob"].isna().any():
        raise ValueError("fpfix_cough_prob contains NaN")
    if not df["fpfix_cough_prob"].between(0.0, 1.0).all():
        raise ValueError("fpfix_cough_prob contains values outside [0, 1]")
    for column in THRESHOLD_COLS:
        df[column] = bool_series(df[column])

    non_cough = df[df["session_type"] == "non_cough"].copy()
    top_noncough = (
        non_cough.sort_values(["scenario_hint", "fpfix_cough_prob"], ascending=[True, False])
        .groupby("scenario_hint", group_keys=False)
        .head(int(args.top_k))
        .copy()
    )
    top_noncough["row_kind"] = "non_cough_top5"
    top_noncough["review_rank"] = top_noncough.groupby("scenario_hint")["fpfix_cough_prob"].rank(
        method="first", ascending=False
    )

    real_cough = df[
        (df["scenario_hint"] == "real_cough") & (df["fpfix_cough_prob"] >= float(args.cough_threshold))
    ].copy()
    real_cough = real_cough.sort_values(["session_key", "window_start_sec"])
    real_cough["row_kind"] = "real_cough_ge_0_50"
    real_cough["review_rank"] = range(1, len(real_cough) + 1)

    columns = [
        "row_kind",
        "review_rank",
        "session_key",
        "session_id",
        "scenario_hint",
        "session_type",
        "source_audio",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "fpfix_cough_prob",
        *THRESHOLD_COLS,
    ]
    optional_columns = [column for column in ["freshness_note", "audio_duration_sec"] if column in df.columns]
    columns.extend(optional_columns)

    top_windows = pd.concat([top_noncough, real_cough], ignore_index=True)
    top_windows["review_rank"] = top_windows["review_rank"].astype(int)
    top_windows = top_windows[columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    top_windows.to_csv(out_path, index=False)

    gate_rows = scenario_gate_summary(df)
    recommendation, recommendation_reason = choose_recommendation(gate_rows)
    write_summary(
        summary_path=summary_path,
        scored_path=scored_path,
        report_path=report_path,
        out_path=out_path,
        df=df,
        top_windows=top_windows,
        gate_rows=gate_rows,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
    )

    print(f"wrote {out_path} rows={len(top_windows)}")
    print(f"wrote {summary_path}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
