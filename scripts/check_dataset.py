from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir


REQUIRED_AUDIO_COLUMNS = {
    "clip_id",
    "session_id",
    "person_id",
    "start_time",
    "end_time",
    "label",
    "audio_file",
}
ALLOWED_LABELS = {"cough", "non_cough"}
ALLOWED_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MIC cough/non_cough dataset metadata and WAV coverage.",
        epilog=(
            "Example:\n"
            "  python scripts/check_dataset.py --labels data/labels/audio_labels.csv "
            "--audio-root data/raw/audio --out results/dataset_report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/audio_baseline.yaml")
    parser.add_argument("--sessions", default=None, help="Path to sessions.csv.")
    parser.add_argument("--labels", default=None, help="Path to audio_labels.csv.")
    parser.add_argument("--audio-root", default=None, help="Directory containing WAV files.")
    parser.add_argument("--out", default=None, help="JSON report path.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for smoke checks.")
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--hop-seconds", type=float, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_report(report: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def wav_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0:
            raise ValueError(f"Invalid samplerate in WAV file: {path}")
        return float(info.frames) / float(info.samplerate)
    except Exception:
        with wave.open(str(path), "rb") as f:
            frames = f.getnframes()
            sample_rate = f.getframerate()
            if sample_rate <= 0:
                raise ValueError(f"Invalid samplerate in WAV file: {path}")
            return float(frames) / float(sample_rate)


def resolve_audio_path(audio_root: Path, audio_file: str | Path) -> Path:
    value = Path(audio_file)
    if value.is_absolute():
        return value
    audio_path = audio_root / value
    root_relative_path = ROOT / value
    if not audio_path.exists() and root_relative_path.exists():
        return root_relative_path
    return audio_path


def window_count(duration: float, window_seconds: float, hop_seconds: float) -> int:
    if duration <= 0:
        return 0
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if duration <= window_seconds:
        return 1
    return int(math.floor((duration - window_seconds) / hop_seconds)) + 1


def empty_report(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    audio = config["audio"]
    return {
        "ok": False,
        "paths": {
            "sessions_csv": str(resolve_path(args.sessions or paths["sessions_csv"])),
            "labels_csv": str(resolve_path(args.labels or paths["labels_csv"])),
            "audio_root": str(resolve_path(args.audio_root or paths["audio_dir"])),
            "out": str(resolve_path(args.out or "results/dataset_report.json")),
        },
        "settings": {
            "max_rows": args.max_rows,
            "window_seconds": float(args.window_seconds or audio["window_seconds"]),
            "hop_seconds": float(args.hop_seconds or audio["hop_seconds"]),
            "allowed_labels": sorted(ALLOWED_LABELS),
        },
        "rows_checked": 0,
        "errors": [],
        "warnings": [],
        "person_id_counts": {},
        "label_stats": {},
        "audio_files": {
            "unique_referenced": 0,
            "found": 0,
            "missing": 0,
            "duration_cache_size": 0,
        },
        "split_check": {
            "status": "not_checked",
            "details": "audio_labels.csv has no split column",
            "person_ids_in_multiple_splits": {},
        },
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = empty_report(args, config)
    paths = report["paths"]
    settings = report["settings"]
    sessions_csv = Path(paths["sessions_csv"])
    labels_csv = Path(paths["labels_csv"])
    audio_root = Path(paths["audio_root"])
    out_path = Path(paths["out"])
    errors: list[str] = report["errors"]
    warnings: list[str] = report["warnings"]

    if not sessions_csv.exists():
        errors.append(f"Missing sessions.csv: {sessions_csv}")
    if not labels_csv.exists():
        errors.append(f"Missing audio_labels.csv: {labels_csv}")
    if not audio_root.exists():
        errors.append(f"Missing audio root directory: {audio_root}")

    labels: pd.DataFrame | None = None
    sessions: pd.DataFrame | None = None
    if labels_csv.exists():
        labels = pd.read_csv(labels_csv)
        if args.max_rows:
            labels = labels.head(args.max_rows)
        report["rows_checked"] = int(len(labels))
    if sessions_csv.exists():
        sessions = pd.read_csv(sessions_csv)

    if labels is None:
        write_report(report, out_path)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Report written to {out_path}")
        return 1

    missing_columns = sorted(REQUIRED_AUDIO_COLUMNS - set(labels.columns))
    if missing_columns:
        errors.append(f"audio_labels.csv missing columns: {missing_columns}")

    if "label" in labels.columns:
        label_values = labels["label"].astype(str)
        invalid_labels = sorted(set(label_values.dropna()) - ALLOWED_LABELS)
        if invalid_labels:
            errors.append(f"Invalid labels found: {invalid_labels}; allowed={sorted(ALLOWED_LABELS)}")

    if "person_id" in labels.columns:
        missing_person = int(labels["person_id"].isna().sum())
        if missing_person:
            errors.append(f"person_id contains {missing_person} empty values")
        report["person_id_counts"] = {
            str(key): int(value)
            for key, value in labels["person_id"].astype(str).value_counts().sort_index().items()
        }

    if sessions is not None and {"session_id"}.issubset(labels.columns) and "session_id" in sessions.columns:
        unknown_sessions = sorted(set(labels["session_id"].dropna()) - set(sessions["session_id"].dropna()))
        if unknown_sessions:
            errors.append(f"Labels reference unknown session_id values: {unknown_sessions[:20]}")

    numeric_start = pd.to_numeric(labels["start_time"], errors="coerce") if "start_time" in labels.columns else None
    numeric_end = pd.to_numeric(labels["end_time"], errors="coerce") if "end_time" in labels.columns else None
    if numeric_start is not None and numeric_end is not None:
        invalid_numeric = labels[numeric_start.isna() | numeric_end.isna()]
        if not invalid_numeric.empty:
            errors.append(f"Found {len(invalid_numeric)} rows with non-numeric start_time/end_time")
        negative_start = labels[numeric_start < 0]
        if not negative_start.empty:
            errors.append(f"Found {len(negative_start)} rows with start_time < 0")
        bad_order = labels[numeric_end <= numeric_start]
        if not bad_order.empty:
            errors.append(f"Found {len(bad_order)} rows with end_time <= start_time")

    duration_cache: dict[str, float] = {}
    found_files = 0
    missing_files = 0
    label_stats: dict[str, dict[str, float | int]] = {
        label: {"rows": 0, "total_duration_seconds": 0.0, "window_count": 0}
        for label in sorted(ALLOWED_LABELS)
    }

    if "audio_file" in labels.columns:
        unique_files = sorted(set(labels["audio_file"].dropna().astype(str)))
        report["audio_files"]["unique_referenced"] = len(unique_files)
        for file_name in unique_files:
            audio_path = resolve_audio_path(audio_root, file_name)
            if not audio_path.exists():
                missing_files += 1
                errors.append(f"Missing audio_file: {audio_path}")
                continue
            try:
                duration_cache[str(audio_path)] = wav_duration_seconds(audio_path)
                found_files += 1
            except Exception as exc:
                errors.append(f"Cannot read WAV duration for {audio_path}: {exc}")

    if {"audio_file", "start_time", "end_time", "label"}.issubset(labels.columns):
        for index, row in labels.iterrows():
            label = str(row["label"])
            start_value = pd.to_numeric(pd.Series([row["start_time"]]), errors="coerce").iloc[0]
            end_value = pd.to_numeric(pd.Series([row["end_time"]]), errors="coerce").iloc[0]
            if pd.isna(start_value) or pd.isna(end_value) or end_value <= start_value:
                continue
            duration = float(end_value) - float(start_value)
            if label in label_stats:
                label_stats[label]["rows"] = int(label_stats[label]["rows"]) + 1
                label_stats[label]["total_duration_seconds"] = round(
                    float(label_stats[label]["total_duration_seconds"]) + duration,
                    6,
                )
                label_stats[label]["window_count"] = int(label_stats[label]["window_count"]) + window_count(
                    duration,
                    float(settings["window_seconds"]),
                    float(settings["hop_seconds"]),
                )

            audio_path = resolve_audio_path(audio_root, str(row["audio_file"]))
            cached_duration = duration_cache.get(str(audio_path))
            if cached_duration is not None and float(end_value) > cached_duration + 1e-6:
                clip_id = row["clip_id"] if "clip_id" in labels.columns else index
                errors.append(
                    f"WAV duration does not cover clip_id={clip_id}: "
                    f"end_time={float(end_value):.3f}s > duration={cached_duration:.3f}s ({audio_path})"
                )

    report["label_stats"] = label_stats
    report["audio_files"]["found"] = found_files
    report["audio_files"]["missing"] = missing_files
    report["audio_files"]["duration_cache_size"] = len(duration_cache)

    if {"person_id", "split"}.issubset(labels.columns):
        split_values = set(labels["split"].dropna().astype(str))
        invalid_splits = sorted(split_values - ALLOWED_SPLITS)
        if invalid_splits:
            errors.append(f"Invalid split values found: {invalid_splits}; allowed={sorted(ALLOWED_SPLITS)}")
        person_splits = labels.groupby("person_id")["split"].nunique(dropna=True)
        leaking_people = person_splits[person_splits > 1].index.tolist()
        leaking_detail = {
            str(person_id): sorted(labels.loc[labels["person_id"] == person_id, "split"].dropna().astype(str).unique())
            for person_id in leaking_people
        }
        report["split_check"] = {
            "status": "failed" if leaking_detail else "passed",
            "details": "Each person_id must appear in only one of train/val/test.",
            "person_ids_in_multiple_splits": leaking_detail,
            "split_counts": {
                str(key): int(value)
                for key, value in labels["split"].astype(str).value_counts().sort_index().items()
            },
        }
        if leaking_detail:
            errors.append(f"person_id appears in multiple splits: {leaking_detail}")
        missing_splits = sorted(ALLOWED_SPLITS - split_values)
        if missing_splits:
            warnings.append(f"Missing split values in labels: {missing_splits}")
    else:
        warnings.append("No split column found; train/val/test grouping by person_id was not checked.")

    report["ok"] = not errors
    write_report(report, out_path)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if errors:
        for error in errors[:30]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 30:
            print(f"ERROR: ... {len(errors) - 30} more errors in {out_path}", file=sys.stderr)
        print(f"Report written to {out_path}")
        return 1

    print(f"OK: checked {len(labels)} rows; report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
