from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


REVIEW_SHEET_COLUMNS = [
    "review_id",
    "review_file",
    "source_audio",
    "source_start_time",
    "source_end_time",
    "clip_start_time",
    "clip_end_time",
    "plan_bucket",
    "decision",
    "review_priority",
    "baseline_cough_prob",
    "yamnet_top1",
    "yamnet_top1_score",
    "yamnet_cough_score",
    "yamnet_sneeze_score",
    "yamnet_speech_score",
    "yamnet_laughter_score",
    "yamnet_breathing_score",
    "yamnet_throat_like_score",
    "reason",
    "manual_decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export WAV clips from a YAMNet review_plan.csv for human listening."
    )
    parser.add_argument(
        "--review-plan",
        default="review/yamnet_full_non_cough_v1/review_plan.csv",
        help="CSV created by scripts/create_yamnet_review_plan.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/human_review/第一次人工重听",
        help="Output folder for copied review clips and review_sheet.csv.",
    )
    parser.add_argument("--context-sec", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--group-by-bucket",
        action="store_true",
        help="Write clips into subfolders by plan_bucket.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing the export script.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing review_sheet.csv and clip files.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def safe_token(value: object, max_len: int = 70) -> str:
    text = str(value)
    text = re.sub(r"\s+", "_", text.strip())
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def prob_token(value: object) -> str:
    try:
        return f"{float(value):.3f}".replace(".", "p")
    except (TypeError, ValueError):
        return "nan"


def time_token(value: object) -> str:
    try:
        return f"{float(value):.2f}".replace(".", "p")
    except (TypeError, ValueError):
        return "nan"


def load_audio(path: Path, sample_rate: int):
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is required. Run this script with .venv_yamnet or install requirements_yamnet.txt."
        ) from exc
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def require_columns(df: pd.DataFrame, path: Path) -> None:
    required = [
        "audio_file",
        "start_time",
        "end_time",
        "plan_bucket",
        "decision",
        "baseline_cough_prob",
        "yamnet_top1",
        "yamnet_top1_score",
        "yamnet_cough_score",
        "yamnet_sneeze_score",
        "yamnet_speech_score",
        "yamnet_laughter_score",
        "yamnet_breathing_score",
        "yamnet_throat_like_score",
        "reason",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def output_filename(index: int, row: pd.Series) -> str:
    bucket = safe_token(row["plan_bucket"], 42)
    prob = prob_token(row["baseline_cough_prob"])
    top1 = safe_token(row["yamnet_top1"], 48)
    start = time_token(row["start_time"])
    end = time_token(row["end_time"])
    return f"{index:05d}_{bucket}_b{prob}_{top1}_{start}_{end}.wav"


def main() -> int:
    args = parse_args()
    if args.context_sec < 0:
        raise ValueError("--context-sec must be non-negative")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    review_plan_path = resolve_path(args.review_plan)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    clips_dir = ensure_dir(out_dir / "clips")
    review_sheet_path = out_dir / "review_sheet.csv"
    errors_path = out_dir / "export_errors.csv"
    report_json_path = out_dir / "export_report.json"

    if not review_plan_path.exists():
        raise FileNotFoundError(f"Review plan not found: {review_plan_path}")
    if review_sheet_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Review sheet already exists: {review_sheet_path}. "
            "Pass --overwrite to replace it."
        )

    plan = pd.read_csv(review_plan_path)
    require_columns(plan, review_plan_path)
    if args.limit is not None:
        plan = plan.head(args.limit).copy()

    audio_cache: dict[Path, object] = {}
    review_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    bucket_counts: dict[str, int] = {}

    for row_index, row in plan.reset_index(drop=True).iterrows():
        review_id = row_index + 1
        source_path = resolve_path(str(row["audio_file"]))
        try:
            if source_path not in audio_cache:
                audio_cache[source_path] = load_audio(source_path, int(args.sample_rate))
            audio = audio_cache[source_path]
            audio_duration = len(audio) / float(args.sample_rate)

            source_start = float(row["start_time"])
            source_end = float(row["end_time"])
            clip_start = max(0.0, source_start - float(args.context_sec))
            clip_end = min(audio_duration, source_end + float(args.context_sec))
            start_sample = int(round(clip_start * args.sample_rate))
            end_sample = int(round(clip_end * args.sample_rate))
            if end_sample <= start_sample:
                raise ValueError(f"Invalid clip range: {clip_start}..{clip_end}")

            bucket = str(row["plan_bucket"])
            target_dir = ensure_dir(clips_dir / safe_token(bucket)) if args.group_by_bucket else clips_dir
            review_path = target_dir / output_filename(review_id, row)
            if review_path.exists() and not args.overwrite:
                raise FileExistsError(f"Clip already exists: {review_path}")

            clip = audio[start_sample:end_sample]
            sf.write(review_path, clip, int(args.sample_rate), subtype="PCM_16")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            review_rows.append(
                {
                    "review_id": f"review_{review_id:05d}",
                    "review_file": display_path(review_path),
                    "source_audio": display_path(source_path),
                    "source_start_time": round(source_start, 6),
                    "source_end_time": round(source_end, 6),
                    "clip_start_time": round(clip_start, 6),
                    "clip_end_time": round(clip_end, 6),
                    "plan_bucket": bucket,
                    "decision": row["decision"],
                    "review_priority": row.get("review_priority", ""),
                    "baseline_cough_prob": row["baseline_cough_prob"],
                    "yamnet_top1": row["yamnet_top1"],
                    "yamnet_top1_score": row["yamnet_top1_score"],
                    "yamnet_cough_score": row["yamnet_cough_score"],
                    "yamnet_sneeze_score": row["yamnet_sneeze_score"],
                    "yamnet_speech_score": row["yamnet_speech_score"],
                    "yamnet_laughter_score": row["yamnet_laughter_score"],
                    "yamnet_breathing_score": row["yamnet_breathing_score"],
                    "yamnet_throat_like_score": row["yamnet_throat_like_score"],
                    "reason": row["reason"],
                    "manual_decision": "",
                    "notes": "",
                }
            )
        except Exception as exc:  # pragma: no cover - depends on local files
            errors.append(
                {
                    "review_id": f"review_{review_id:05d}",
                    "audio_file": str(row.get("audio_file", "")),
                    "plan_bucket": str(row.get("plan_bucket", "")),
                    "start_time": row.get("start_time", ""),
                    "end_time": row.get("end_time", ""),
                    "error": str(exc),
                }
            )

    pd.DataFrame(review_rows, columns=REVIEW_SHEET_COLUMNS).to_csv(review_sheet_path, index=False)
    pd.DataFrame(errors).to_csv(errors_path, index=False)

    report = {
        "status": "completed",
        "review_plan": str(review_plan_path),
        "out_dir": str(out_dir),
        "clips_dir": str(clips_dir),
        "review_sheet": str(review_sheet_path),
        "errors_csv": str(errors_path),
        "report_json": str(report_json_path),
        "plan_rows": int(len(plan)),
        "exported_clips": int(len(review_rows)),
        "error_rows": int(len(errors)),
        "source_audio_count": int(plan["audio_file"].nunique()) if not plan.empty else 0,
        "bucket_counts": bucket_counts,
        "context_sec": float(args.context_sec),
        "sample_rate": int(args.sample_rate),
        "manual_decision_values": [
            "cough",
            "hard_negative",
            "clean_non_cough",
            "uncertain",
            "exclude",
        ],
        "next_step": "Listen to clips and fill review_sheet.csv manual_decision and notes. Do not build labels until review is complete.",
    }
    report_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("COPY_FOR_CODEX_BEGIN")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("COPY_FOR_CODEX_END")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
