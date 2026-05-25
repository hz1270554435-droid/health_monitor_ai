from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an approximate duration-limited WAV subset for YAMNet hard mining."
    )
    parser.add_argument("--source-dir", default="D:/cough_model_train/DATA/non_cough")
    parser.add_argument("--out-dir", default="DATA/hard_data")
    parser.add_argument("--target-sec", type=float, default=8000.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def safe_relative_path(source_root: Path, wav_path: Path) -> Path:
    try:
        rel_path = wav_path.relative_to(source_root)
    except ValueError:
        digest = hashlib.sha1(str(wav_path).encode("utf-8")).hexdigest()[:10]
        rel_path = Path(f"{wav_path.stem}_{digest}{wav_path.suffix}")
    return rel_path


def iter_wavs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")


def main() -> int:
    args = parse_args()
    if args.target_sec <= 0:
        raise ValueError("--target-sec must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when provided")

    source_dir = resolve_path(args.source_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    manifest_path = resolve_path(args.manifest) if args.manifest else out_dir / "hard_data_manifest.csv"
    ensure_dir(manifest_path.parent)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    total_sec = 0.0
    copied = 0
    for wav_path in iter_wavs(source_dir):
        if args.max_files is not None and copied >= args.max_files:
            break
        if total_sec >= args.target_sec:
            break
        try:
            info = sf.info(str(wav_path))
            duration = float(info.frames) / float(info.samplerate)
        except Exception as exc:  # pragma: no cover - depends on local files
            errors.append({"source_audio": str(wav_path), "error": str(exc)})
            continue

        rel_path = safe_relative_path(source_dir, wav_path)
        target_path = out_dir / rel_path
        ensure_dir(target_path.parent)
        shutil.copy2(wav_path, target_path)
        copied += 1
        total_sec += duration
        rows.append(
            {
                "source_audio": str(wav_path.resolve()),
                "copied_audio": display_path(target_path),
                "duration": duration,
                "cumulative_duration": total_sec,
            }
        )

    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "hard_data_copy_errors.csv", index=False)
    print(
        f"OK: copied {copied} wav files, cumulative_duration={total_sec:.2f}s, "
        f"manifest={manifest_path}"
    )
    if errors:
        print(f"WARN: {len(errors)} files could not be inspected; see {out_dir / 'hard_data_copy_errors.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
