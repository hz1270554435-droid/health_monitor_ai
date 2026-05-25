from __future__ import annotations

import argparse
import csv
import shutil
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a balanced cough/non_cough audio subset.")
    parser.add_argument("--config", default="configs/audio_formal_50.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate in WAV: {path}")
        return frames / float(sample_rate)


def wav_sample_rate(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return int(wav.getframerate())


def find_wavs(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    wavs = []
    for path in source_dir.rglob("*.wav"):
        if "_reports" in path.parts:
            continue
        wavs.append(path)
    return sorted(wavs)


def select_wavs(source_dir: Path, count: int, sample_rate: int, min_duration: float) -> list[Path]:
    selected = []
    for wav_path in find_wavs(source_dir):
        try:
            if wav_sample_rate(wav_path) != sample_rate:
                continue
            if wav_duration_seconds(wav_path) < min_duration:
                continue
        except Exception:
            continue
        selected.append(wav_path)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only found {len(selected)} usable WAV files in {source_dir}; need {count}")


def class_split(index: int, count: int, train_ratio: float, val_ratio: float) -> str:
    train_count = max(1, round(count * train_ratio))
    val_count = max(1, round(count * val_ratio))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    if index < train_count:
        return "train"
    if index < train_count + val_count:
        return "val"
    return "test"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    subset = config["dataset_subset"]
    paths = config["paths"]
    labels_cfg = config["labels"]
    split_cfg = config["split"]
    sample_rate = int(config["audio"]["sample_rate"])

    output_audio_dir = resolve_path(paths["audio_dir"])
    raw_dir = ROOT / "data" / "raw"
    try:
        output_audio_dir.resolve().relative_to(raw_dir.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(f"Refusing to write subset audio under data/raw: {output_audio_dir}")

    labels_csv = resolve_path(paths["labels_csv"])
    sessions_csv = resolve_path(paths["sessions_csv"])
    if not args.overwrite:
        existing_targets = [path for path in (output_audio_dir, labels_csv, sessions_csv) if path.exists()]
        if existing_targets:
            raise FileExistsError(f"Outputs already exist; pass --overwrite to replace: {existing_targets}")

    per_class = int(subset["per_class"])
    min_duration = float(subset.get("min_duration_seconds", 1.0))
    classes = [
        (labels_cfg["positive"], resolve_path(subset["cough_source_dir"])),
        (labels_cfg["negative"], resolve_path(subset["non_cough_source_dir"])),
    ]

    ensure_dir(output_audio_dir)
    sessions: list[dict] = []
    labels: list[dict] = []
    person_index = 1

    for label, source_dir in classes:
        selected = select_wavs(source_dir, per_class, sample_rate, min_duration)
        for class_index, wav_path in enumerate(selected, start=1):
            person_id = f"P_FORMAL_{person_index:03d}"
            split = class_split(
                class_index - 1,
                per_class,
                float(split_cfg["train_ratio"]),
                float(split_cfg["val_ratio"]),
            )
            stem = f"{label}_{class_index:04d}"
            target_name = f"{stem}.wav"
            target_path = output_audio_dir / target_name
            if target_path.exists() and not args.overwrite:
                raise FileExistsError(f"Output WAV already exists: {target_path}")
            shutil.copy2(wav_path, target_path)

            duration = wav_duration_seconds(target_path)
            audio_file = target_path.relative_to(ROOT).as_posix()
            session_id = f"session_{stem}"
            clip_id = f"clip_{stem}"
            row_common = {
                "session_id": session_id,
                "person_id": person_id,
                "start_time": "0.000000",
                "end_time": f"{duration:.6f}",
                "label": label,
                "audio_file": audio_file,
                "split": split,
            }
            sessions.append(
                {
                    **row_common,
                    "source": str(wav_path),
                }
            )
            labels.append(
                {
                    "clip_id": clip_id,
                    **row_common,
                }
            )
            person_index += 1

    write_csv(
        sessions_csv,
        sessions,
        ["session_id", "person_id", "source", "label", "audio_file", "start_time", "end_time", "split"],
    )
    write_csv(
        labels_csv,
        labels,
        ["clip_id", "session_id", "person_id", "start_time", "end_time", "label", "audio_file", "split"],
    )
    print(f"OK: copied {len(labels)} WAV files to {output_audio_dir}")
    print(f"OK: wrote labels to {labels_csv}")
    print(f"OK: wrote sessions to {sessions_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
