from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import load_audio_mono, resolve_path, stable_id


MANUAL_COLUMNS = ["manual_decision", "manual_subtype", "manual_notes"]
REQUIRED_COLUMNS = [
    "session_id",
    "person_id",
    "review_bucket",
    "v3_cough_prob",
    "yamnet_top1",
]
REQUIRED_OUTPUT_FIELDS = [
    "original_path",
    "session_id",
    "person_id",
    "window_start_sec",
    "window_end_sec",
    "review_bucket",
    "v3_cough_prob",
    "yamnet_top1",
]
MANUAL_DECISION_VALUES = ["cough", "hard_negative", "clean_non_cough", "uncertain", "exclude"]
MANUAL_SUBTYPE_SUGGESTIONS = [
    "clear_cough",
    "cough_with_speech",
    "speech",
    "loud_speech",
    "throat_clear",
    "laugh",
    "breath",
    "snort",
    "room_noise",
    "silence",
    "handling_noise",
    "clipping",
    "unknown",
]
NOT_EXECUTED = [
    "build_v3_1_candidate_from_manual_reviews.py",
    "audio_labels_v3_1_candidate.csv generation",
    "training",
    "raw audio copy",
    "ml/data/raw modification",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v3.1 review clips and review_sheet.csv.")
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--review-plan", default=None)
    parser.add_argument("--out-dir", default=None, help="Legacy output dir; used when explicit outputs are omitted.")
    parser.add_argument("--clips-dir", default=None)
    parser.add_argument("--review-sheet", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--context-sec", type=float, default=0.5, help="Legacy symmetric context seconds.")
    parser.add_argument("--pre-context-sec", type=float, default=None)
    parser.add_argument("--post-context-sec", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--group-by-bucket", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def path_text(path: Path) -> str:
    return str(path.resolve())


def relpath_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def clean_filename_token(value: object, max_len: int = 80) -> str:
    text = str(value).strip()
    text = re.sub(r'[\\/:*?"<>|,\s]+', "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return (text or "unknown")[:max_len]


def format_float_token(value: object, digits: int) -> str:
    number = float(value)
    return f"{number:.{digits}f}"


def get_audio_path(row: pd.Series) -> Path:
    for column in ("original_path", "audio_file"):
        value = str(row.get(column, "")).strip()
        if value:
            return resolve_path(value)
    raise ValueError("Row has neither original_path nor audio_file")


def get_window_times(row: pd.Series) -> tuple[float, float]:
    start_value = row.get("window_start_sec", row.get("start_time", ""))
    end_value = row.get("window_end_sec", row.get("end_time", ""))
    start = float(start_value)
    end = float(end_value)
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"Invalid window range: {start_value}..{end_value}")
    return start, end


def clip_filename(index: int, row: pd.Series) -> str:
    bucket = clean_filename_token(row["review_bucket"], 48)
    session_id = clean_filename_token(row["session_id"], 32)
    start = clean_filename_token(format_float_token(row.get("window_start_sec", row.get("start_time")), 2))
    prob = clean_filename_token(format_float_token(row.get("v3_cough_prob", 0.0), 3))
    top1 = clean_filename_token(row.get("yamnet_top1", "yamnet"), 36)
    return f"{index:05d}_{bucket}_{session_id}_start{start}_v3{prob}_{top1}.wav"


def require_columns(frame: pd.DataFrame, path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if "original_path" not in frame.columns and "audio_file" not in frame.columns:
        missing.append("original_path or audio_file")
    if "window_start_sec" not in frame.columns and "start_time" not in frame.columns:
        missing.append("window_start_sec or start_time")
    if "window_end_sec" not in frame.columns and "end_time" not in frame.columns:
        missing.append("window_end_sec or end_time")
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def manual_columns_nonempty(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in MANUAL_COLUMNS:
        if column in frame.columns:
            counts[column] = int((frame[column].astype(str).str.strip() != "").sum())
        else:
            counts[column] = 0
    return counts


def guard_existing_review_sheet(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Review sheet exists; pass --overwrite to replace blank sheets: {path}")
    existing = pd.read_csv(path, keep_default_na=False)
    nonempty = manual_columns_nonempty(existing)
    dirty = {column: count for column, count in nonempty.items() if count}
    if dirty:
        raise RuntimeError(
            "Refusing to overwrite review_sheet.csv because manual review fields are non-empty: "
            f"{dirty}. This stage intentionally has no --overwrite-manual option."
        )


def source_base_dir(paths: list[Path]) -> Path:
    parents = [str(path.resolve().parent) for path in paths]
    if not parents:
        return ROOT
    return Path(os.path.commonpath(parents))


def expected_clip_path(clips_dir: Path, index: int, row: pd.Series, group_by_bucket: bool) -> Path:
    target_dir = clips_dir / clean_filename_token(row["review_bucket"]) if group_by_bucket else clips_dir
    return target_dir / clip_filename(index, row)


def guard_existing_clips(clips_dir: Path, expected_paths: set[Path], overwrite: bool) -> None:
    if not clips_dir.exists():
        return
    existing_wavs = {path.resolve() for path in clips_dir.rglob("*.wav")}
    if not existing_wavs:
        return
    expected_resolved = {path.resolve() for path in expected_paths}
    stale = sorted(existing_wavs - expected_resolved)
    if stale:
        preview = [str(path) for path in stale[:10]]
        raise RuntimeError(
            f"Refusing to export because {len(stale)} unexpected .wav files already exist under {clips_dir}: {preview}"
        )
    if not overwrite:
        raise FileExistsError(f"Review clips already exist; pass --overwrite to replace expected blank-export clips: {clips_dir}")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Stage v3.1-EXPORTCLIPS-1 Summary",
        "",
        "This stage exports short clips and a blank review sheet for human listening. It does not generate "
        "candidate labels, train a model, copy full sessions, or modify ml/data/raw/.",
        "",
        "## Outputs",
        "",
        f"- review_plan: `{report['review_plan']}`",
        f"- review_sheet: `{report['review_sheet']}`",
        f"- clips_dir: `{report['clips_dir']}`",
        f"- report_json: `{report['report_json']}`",
        "",
        "## Counts",
        "",
        f"- plan_rows: `{report['plan_rows']}`",
        f"- exported_clips: `{report['exported_clips']}`",
        f"- failed_rows: `{report['failed_rows']}`",
        f"- boundary_clipped_count: `{report['boundary_clipped_count']}`",
        "",
        "## Audio Handling",
        "",
        f"- sample_rate: `{report['sample_rate']}`",
        f"- input_channels: `{report['input_channels']}`",
        f"- export_channels: `{report['export_channels']}`",
        f"- audio_mix_mode: `{report['audio_mix_mode']}`",
        "",
        "## Clip Counts Per Bucket",
        "",
    ]
    for key, value in report["clip_counts_per_bucket"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Clip Counts Per Session", ""])
    for key, value in report["clip_counts_per_session"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Manual Review Fields",
            "",
            "- `manual_decision`, `manual_subtype`, and `manual_notes` are present and intentionally blank.",
            f"- manual_decision_allowed_values: `{', '.join(report['manual_decision_allowed_values'])}`",
            f"- manual_subtype_suggestions: `{', '.join(report['manual_subtype_suggestions'])}`",
            "",
            "## Validation",
            "",
            f"- wav_readable_count: `{report['wav_validation']['readable_count']}`",
            f"- wav_validation_failed_count: `{report['wav_validation']['failed_count']}`",
            f"- max_duration_abs_error_sec: `{report['wav_validation']['max_duration_abs_error_sec']}`",
            "",
            "## Not Executed",
            "",
        ]
    )
    for item in report["not_executed"]:
        lines.append(f"- {item}")
    if report["failed_rows_detail"]:
        lines.extend(["", "## Failed Rows", ""])
        for row in report["failed_rows_detail"]:
            lines.append(f"- row {row.get('review_index')}: {row.get('error')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def validate_review_sheet(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in REQUIRED_OUTPUT_FIELDS if column not in frame.columns]
    if missing:
        raise ValueError(f"review_sheet.csv is missing required preserved fields: {missing}")
    empty_required = {
        column: int((frame[column].astype(str).str.strip() == "").sum()) for column in REQUIRED_OUTPUT_FIELDS
    }
    nonempty_manual = manual_columns_nonempty(frame)
    dirty_manual = {column: count for column, count in nonempty_manual.items() if count}
    if dirty_manual:
        raise ValueError(f"Manual review columns must be empty after export: {dirty_manual}")
    return {
        "required_empty_counts": empty_required,
        "manual_nonempty_counts": nonempty_manual,
    }


def main() -> int:
    args = parse_args()
    pre_context = float(args.pre_context_sec if args.pre_context_sec is not None else args.context_sec)
    post_context = float(args.post_context_sec if args.post_context_sec is not None else args.context_sec)
    if pre_context < 0 or post_context < 0:
        raise ValueError("Context seconds must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    config = load_config(args.config)
    paths = config["paths"]
    review_plan_path = resolve_path(args.review_plan or paths["review_plan_csv"])
    out_dir = resolve_path(args.out_dir or paths["review_dir"])
    clips_dir = resolve_path(args.clips_dir) if args.clips_dir else out_dir / "clips"
    review_sheet_path = resolve_path(args.review_sheet) if args.review_sheet else out_dir / "review_sheet.csv"
    report_path = resolve_path(args.report) if args.report else out_dir / "export_report.json"
    summary_path = resolve_path(args.summary_md) if args.summary_md else out_dir / "export_summary.md"
    sample_rate = int(args.sample_rate or config["audio"].get("sample_rate", 16000))
    pcm_defaults = dict(config.get("mining", {}).get("pcm_defaults", {}))

    if not review_plan_path.exists():
        raise FileNotFoundError(f"Review plan not found: {review_plan_path}")
    for path in (report_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace: {path}")
    guard_existing_review_sheet(review_sheet_path, args.overwrite)

    plan = pd.read_csv(review_plan_path, keep_default_na=False)
    require_columns(plan, review_plan_path)
    input_manual_nonempty = manual_columns_nonempty(plan)
    dirty_input_manual = {column: count for column, count in input_manual_nonempty.items() if count}
    if dirty_input_manual:
        raise ValueError(f"Review plan manual columns must be empty before export: {dirty_input_manual}")
    if args.limit is not None:
        plan = plan.head(args.limit).copy()

    source_paths = [get_audio_path(row) for _, row in plan.iterrows()]
    source_base = source_base_dir(source_paths)
    expected_paths = {
        expected_clip_path(clips_dir, index, row, args.group_by_bucket)
        for index, (_, row) in enumerate(plan.reset_index(drop=True).iterrows(), start=1)
    }
    guard_existing_clips(clips_dir, expected_paths, args.overwrite)

    ensure_dir(clips_dir)
    ensure_dir(review_sheet_path.parent)
    ensure_dir(report_path.parent)
    ensure_dir(summary_path.parent)

    audio_cache: dict[Path, Any] = {}
    info_cache: dict[Path, sf.SoundFile] = {}
    review_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    wav_checks: list[dict[str, object]] = []
    boundary_clipped_count = 0
    duration_errors: list[float] = []
    input_channels: set[int] = set()
    input_sample_rates: set[int] = set()

    for review_index, row in enumerate(plan.reset_index(drop=True).iterrows(), start=1):
        _, row_data = row
        source_path = get_audio_path(row_data)
        source_start, source_end = get_window_times(row_data)
        clip_id = stable_id("v3_1_review", source_path, source_start, source_end, row_data["review_bucket"])
        clip_path = expected_clip_path(clips_dir, review_index, row_data, args.group_by_bucket)
        try:
            source_info = info_cache.get(source_path)
            if source_info is None:
                source_info = sf.info(str(source_path))
                info_cache[source_path] = source_info
            input_channels.add(int(source_info.channels))
            input_sample_rates.add(int(source_info.samplerate))

            if source_path not in audio_cache:
                audio_cache[source_path] = load_audio_mono(source_path, sample_rate, pcm_defaults)
            audio = audio_cache[source_path]
            audio_duration = len(audio) / float(sample_rate)
            desired_start = source_start - pre_context
            desired_end = source_end + post_context
            clip_start = max(0.0, desired_start)
            clip_end = min(audio_duration, desired_end)
            if clip_start != desired_start or clip_end != desired_end:
                boundary_clipped_count += 1
            start_sample = int(round(clip_start * sample_rate))
            end_sample = int(round(clip_end * sample_rate))
            if end_sample <= start_sample:
                raise ValueError(f"Invalid clip range: {clip_start}..{clip_end}")

            ensure_dir(clip_path.parent)
            sf.write(clip_path, audio[start_sample:end_sample], sample_rate, subtype="PCM_16")
            clip_info = sf.info(str(clip_path))
            actual_duration = float(clip_info.frames) / float(clip_info.samplerate)
            expected_duration = float(end_sample - start_sample) / float(sample_rate)
            duration_error = abs(actual_duration - expected_duration)
            duration_errors.append(duration_error)
            validation_error = ""
            if clip_info.samplerate != sample_rate:
                validation_error = f"sample_rate={clip_info.samplerate}"
            elif clip_info.channels != 1:
                validation_error = f"channels={clip_info.channels}"
            elif actual_duration <= 0:
                validation_error = "duration<=0"
            elif duration_error > max(1.0 / sample_rate, 1e-4):
                validation_error = f"duration_error={duration_error}"
            wav_checks.append(
                {
                    "clip_path": path_text(clip_path),
                    "readable": validation_error == "",
                    "sample_rate": int(clip_info.samplerate),
                    "channels": int(clip_info.channels),
                    "duration_sec": round(actual_duration, 6),
                    "expected_duration_sec": round(expected_duration, 6),
                    "duration_abs_error_sec": round(duration_error, 9),
                    "error": validation_error,
                }
            )
            if validation_error:
                raise ValueError(f"WAV validation failed: {validation_error}")

            base_row = row_data.to_dict()
            for column in MANUAL_COLUMNS:
                base_row.pop(column, None)
            base_row.update(
                {
                    "clip_id": clip_id,
                    "clip_path": path_text(clip_path),
                    "clip_relpath": relpath_or_abs(clip_path, ROOT),
                    "source_audio": path_text(source_path),
                    "source_relpath": relpath_or_abs(source_path, source_base),
                    "clip_start_sec": round(clip_start, 6),
                    "clip_end_sec": round(clip_end, 6),
                    "clip_duration_sec": round(actual_duration, 6),
                    "manual_decision": "",
                    "manual_subtype": "",
                    "manual_notes": "",
                }
            )
            review_rows.append(base_row)
        except Exception as exc:
            failed_rows.append(
                {
                    "review_index": review_index,
                    "clip_id": clip_id,
                    "source_audio": path_text(source_path),
                    "session_id": row_data.get("session_id", ""),
                    "review_bucket": row_data.get("review_bucket", ""),
                    "window_start_sec": row_data.get("window_start_sec", row_data.get("start_time", "")),
                    "window_end_sec": row_data.get("window_end_sec", row_data.get("end_time", "")),
                    "error": str(exc),
                }
            )

    sheet = pd.DataFrame(review_rows)
    if sheet.empty:
        sheet = pd.DataFrame(columns=list(plan.columns) + [
            "clip_id",
            "clip_path",
            "clip_relpath",
            "source_audio",
            "source_relpath",
            "clip_start_sec",
            "clip_end_sec",
            "clip_duration_sec",
            *MANUAL_COLUMNS,
        ])
    validation = validate_review_sheet(sheet)
    sheet.to_csv(review_sheet_path, index=False, encoding="utf-8-sig")

    failed_wav_checks = [item for item in wav_checks if not item["readable"]]
    clip_duration_values = [float(item["duration_sec"]) for item in wav_checks if item["readable"]]
    short_boundary_clips = [item for item in wav_checks if item["readable"] and float(item["duration_sec"]) < 2.0]
    report = {
        "stage": "v3.1-EXPORTCLIPS-1",
        "review_plan": path_text(review_plan_path),
        "review_sheet": path_text(review_sheet_path),
        "clips_dir": path_text(clips_dir),
        "report_json": path_text(report_path),
        "summary_md": path_text(summary_path),
        "source_audio_base_dir": path_text(source_base),
        "plan_rows": int(len(plan)),
        "exported_clips": int(len(review_rows)),
        "failed_rows": int(len(failed_rows)),
        "failed_rows_detail": failed_rows,
        "pre_context_sec": pre_context,
        "post_context_sec": post_context,
        "sample_rate": sample_rate,
        "input_channels": sorted(input_channels),
        "input_sample_rates": sorted(input_sample_rates),
        "export_channels": 1,
        "audio_mix_mode": "librosa_mono_average",
        "clip_counts_per_bucket": counts(sheet["review_bucket"]) if "review_bucket" in sheet else {},
        "clip_counts_per_session": counts(sheet["session_id"]) if "session_id" in sheet else {},
        "boundary_clipped_count": int(boundary_clipped_count),
        "short_boundary_clip_count": int(len(short_boundary_clips)),
        "clip_duration_summary_sec": {
            "min": round(min(clip_duration_values), 6) if clip_duration_values else None,
            "mean": round(sum(clip_duration_values) / len(clip_duration_values), 6) if clip_duration_values else None,
            "max": round(max(clip_duration_values), 6) if clip_duration_values else None,
        },
        "wav_validation": {
            "readable_count": int(len(wav_checks) - len(failed_wav_checks)),
            "failed_count": int(len(failed_wav_checks)),
            "max_duration_abs_error_sec": round(max(duration_errors), 9) if duration_errors else None,
            "failed_detail": failed_wav_checks,
        },
        "review_sheet_validation": validation,
        "manual_decision_allowed_values": MANUAL_DECISION_VALUES,
        "manual_subtype_suggestions": MANUAL_SUBTYPE_SUGGESTIONS,
        "manual_columns_blank": True,
        "not_executed": NOT_EXECUTED,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not failed_rows and not failed_wav_checks else 2


if __name__ == "__main__":
    raise SystemExit(main())
