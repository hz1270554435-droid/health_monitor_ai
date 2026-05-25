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

from src.common.io import ensure_dir
from v3_1_mining_common import audio_info, display_path, iter_window_ranges, stable_id


MANIFEST_COLUMNS = [
    "original_path",
    "audio_file",
    "metadata_json",
    "session_id",
    "person_id",
    "source_domain",
    "scenario_hint",
    "fs",
    "channels",
    "duration_s",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "start_time",
    "end_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a window-level manifest from board-live 1min session WAV/JSON pairs.")
    parser.add_argument("--session-dir", required=True, help="Directory containing *_audio.wav and matching *_audio.json files.")
    parser.add_argument("--out", default="review/board_live_data_1/window_manifest.csv")
    parser.add_argument("--source-domain", default="board_live_data_1")
    parser.add_argument("--scenario-hint", default="unknown")
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--limit-sessions", type=int, default=None, help="Optional smoke-test limiter.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def iter_session_wavs(session_dir: Path) -> list[Path]:
    return sorted(path for path in session_dir.glob("*_audio.wav") if path.is_file())


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata JSON for session audio: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def value_or_default(metadata: dict[str, Any], name: str, default: object) -> object:
    value = metadata.get(name, default)
    return default if value is None or value == "" else value


def build_rows(
    session_dir: Path,
    source_domain: str,
    scenario_hint: str,
    window_sec: float,
    hop_sec: float,
    limit_sessions: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("--window-sec and --hop-sec must be positive")
    if limit_sessions is not None and limit_sessions <= 0:
        raise ValueError("--limit-sessions must be positive when provided")

    wavs = iter_session_wavs(session_dir)
    if limit_sessions is not None:
        wavs = wavs[:limit_sessions]

    rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    sessions = 0

    for wav_path in wavs:
        json_path = wav_path.with_suffix(".json")
        try:
            metadata = load_metadata(json_path)
            info = audio_info(wav_path, {})
            fs = int(value_or_default(metadata, "fs", info["sample_rate"]))
            channels = int(value_or_default(metadata, "channels", info["channels"]))
            duration_s = float(value_or_default(metadata, "duration_s", info["duration_sec"]))
            audio_size = int(round(duration_s * fs))
            session_id = str(value_or_default(metadata, "session_id", wav_path.stem.replace("_audio", "")))
            person_id = str(value_or_default(metadata, "person_id", "unknown_person"))
            session_scenario = str(value_or_default(metadata, "scene", scenario_hint)) if scenario_hint == "unknown" else scenario_hint
            original_path = display_path(wav_path)
            metadata_json = display_path(json_path)

            for index, (start_sample, end_sample) in enumerate(iter_window_ranges(audio_size, fs, window_sec, hop_sec)):
                start_time = round(start_sample / float(fs), 6)
                end_time = round(min(end_sample, audio_size) / float(fs), 6)
                window_id = stable_id("win", original_path, session_id, start_time, end_time, length=12)
                rows.append(
                    {
                        "original_path": original_path,
                        "audio_file": original_path,
                        "metadata_json": metadata_json,
                        "session_id": session_id,
                        "person_id": person_id,
                        "source_domain": source_domain,
                        "scenario_hint": session_scenario,
                        "fs": fs,
                        "channels": channels,
                        "duration_s": round(duration_s, 6),
                        "window_id": window_id,
                        "window_start_sec": start_time,
                        "window_end_sec": end_time,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
            sessions += 1
        except Exception as exc:
            errors.append({"audio_file": str(wav_path), "metadata_json": str(json_path), "error": str(exc)})

    report = {
        "session_dir": str(session_dir),
        "sessions_scanned": sessions,
        "windows": len(rows),
        "error_rows": len(errors),
        "errors": errors[:50],
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "source_domain": source_domain,
        "scenario_hint_default": scenario_hint,
    }
    return rows, report


def main() -> int:
    args = parse_args()
    session_dir = resolve_path(args.session_dir)
    out_path = resolve_path(args.out)
    report_path = out_path.with_name(f"{out_path.stem}_report.json")

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")

    rows, report = build_rows(
        session_dir=session_dir,
        source_domain=str(args.source_domain),
        scenario_hint=str(args.scenario_hint),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        limit_sessions=args.limit_sessions,
    )
    ensure_dir(out_path.parent)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(out_path, index=False)
    report.update({"manifest_csv": str(out_path), "report_json": str(report_path)})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["error_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
