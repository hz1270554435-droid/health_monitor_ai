from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


REVIEW_COLUMNS = [
    "review_file",
    "source_audio",
    "start_time",
    "end_time",
    "decision",
    "review_priority",
    "baseline_cough_prob",
    "yamnet_top1",
    "yamnet_top1_score",
    "yamnet_cough_score",
    "yamnet_sneeze_score",
    "yamnet_speech_score",
    "manual_decision",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export short WAV clips for human YAMNet hard-mining review.")
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-hard", type=int, default=300)
    parser.add_argument("--top-baseline-suspicious", type=int, default=300)
    parser.add_argument("--top-pseudo-cough", type=int, default=200)
    parser.add_argument("--top-uncertain", type=int, default=200)
    parser.add_argument("--context-sec", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def safe_token(value: object) -> str:
    text = str(value)
    cleaned = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text]
    return "".join(cleaned).strip("_")[:64] or "unknown"


def prob_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def time_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def load_audio(path: Path, sample_rate: int):
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for review clip export.") from exc
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = []

    hard = df[df["decision"] == "hard_negative_review"].copy()
    if not hard.empty:
        hard = hard.sort_values("baseline_cough_prob", ascending=False).head(args.top_hard)
        selected.append(hard)

    suspicious = df[df["decision"] == "baseline_suspicious_review"].copy()
    if not suspicious.empty:
        suspicious = suspicious.sort_values("baseline_cough_prob", ascending=False).head(
            args.top_baseline_suspicious
        )
        selected.append(suspicious)

    pseudo = df[df["decision"] == "pseudo_cough_candidate"].copy()
    if not pseudo.empty:
        pseudo["rank_score"] = pseudo["baseline_cough_prob"] + pseudo["yamnet_cough_score"]
        pseudo = pseudo.sort_values("rank_score", ascending=False).head(args.top_pseudo_cough)
        selected.append(pseudo)

    uncertain = df[df["decision"] == "uncertain_review"].copy()
    if not uncertain.empty:
        uncertain["rank_score"] = (uncertain["baseline_cough_prob"] - float(args.threshold)).abs()
        uncertain = uncertain.sort_values("rank_score", ascending=True).head(args.top_uncertain)
        selected.append(uncertain)

    if not selected:
        return pd.DataFrame(columns=df.columns)
    output = pd.concat(selected, ignore_index=True)
    if "sample_id" in output.columns:
        output = output.drop_duplicates(subset=["sample_id"], keep="first")
    return output.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    if args.context_sec < 0:
        raise ValueError("--context-sec must be non-negative")
    candidate_csv = resolve_path(args.candidate_csv)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    clips_dir = ensure_dir(out_dir / "clips")
    review_sheet_path = out_dir / "review_sheet.csv"
    errors_path = out_dir / "export_errors.csv"

    if not candidate_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")
    candidates = pd.read_csv(candidate_csv)
    selected = select_rows(candidates, args)

    review_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    audio_cache: dict[Path, object] = {}

    for index, row in selected.iterrows():
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
                raise ValueError(f"Invalid export range: {clip_start}..{clip_end}")

            clip = audio[start_sample:end_sample]
            filename = (
                f"{index + 1:05d}_{safe_token(row['decision'])}_"
                f"b{prob_token(float(row['baseline_cough_prob']))}_"
                f"{safe_token(row['yamnet_top1'])}_"
                f"{time_token(source_start)}_{time_token(source_end)}.wav"
            )
            review_path = clips_dir / filename
            sf.write(review_path, clip, int(args.sample_rate), subtype="PCM_16")
            review_rows.append(
                {
                    "review_file": display_path(review_path),
                    "source_audio": display_path(source_path),
                    "start_time": round(clip_start, 6),
                    "end_time": round(clip_end, 6),
                    "decision": row["decision"],
                    "review_priority": row.get("review_priority", ""),
                    "baseline_cough_prob": row["baseline_cough_prob"],
                    "yamnet_top1": row["yamnet_top1"],
                    "yamnet_top1_score": row["yamnet_top1_score"],
                    "yamnet_cough_score": row["yamnet_cough_score"],
                    "yamnet_sneeze_score": row["yamnet_sneeze_score"],
                    "yamnet_speech_score": row["yamnet_speech_score"],
                    "manual_decision": "",
                    "notes": "",
                }
            )
        except Exception as exc:  # pragma: no cover - depends on local files
            errors.append(
                {
                    "audio_file": str(row.get("audio_file", "")),
                    "sample_id": str(row.get("sample_id", "")),
                    "error": str(exc),
                }
            )

    pd.DataFrame(review_rows, columns=REVIEW_COLUMNS).to_csv(review_sheet_path, index=False)
    pd.DataFrame(errors, columns=["audio_file", "sample_id", "error"]).to_csv(errors_path, index=False)
    print(
        f"OK: exported {len(review_rows)} clips to {clips_dir}; "
        f"review_sheet={review_sheet_path}; errors={len(errors)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
