from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import iter_windows, load_audio_segment, log_mel_feature
from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Log-Mel features for an audio binary task.")
    parser.add_argument("--config", default="configs/audio_baseline.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def row_value(row, names: tuple[str, ...], default=None):
    for name in names:
        if hasattr(row, name):
            value = getattr(row, name)
            if value == value:
                return value
    return default


def resolve_audio_path(root: Path, audio_root: Path, audio_value: object) -> Path:
    audio_file = Path(str(audio_value))
    if audio_file.is_absolute():
        return audio_file
    audio_path = audio_root / audio_file
    root_relative_path = root / audio_file
    if not audio_path.exists() and root_relative_path.exists():
        return root_relative_path
    return audio_path


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    audio_config = config["audio"]
    labels_csv = paths.get("task_view_csv") or paths["labels_csv"]
    labels_path = Path(labels_csv)
    if not labels_path.is_absolute():
        labels_path = ROOT / labels_path
    labels = pd.read_csv(labels_path)
    if args.limit:
        labels = labels.head(args.limit)

    output_dir = ensure_dir(args.output_dir or (ROOT / paths["processed_dir"]))
    feature_dir = ensure_dir(output_dir / "features")
    manifest_path = output_dir / "audio_manifest.csv"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest exists; pass --overwrite to replace: {manifest_path}")

    label_map = {
        config["labels"]["negative"]: 0,
        config["labels"]["positive"]: 1,
    }
    rows: list[dict] = []
    audio_dir = Path(paths.get("audio_dir", "."))
    if not audio_dir.is_absolute():
        audio_dir = ROOT / audio_dir
    sample_rate = int(audio_config["sample_rate"])

    for label_row in labels.itertuples(index=False):
        label = str(row_value(label_row, ("task_label", "label")))
        if label not in label_map:
            raise ValueError(
                f"Unsupported label for clip_id={label_row.clip_id}: {label}; "
                f"expected one of {sorted(label_map)}"
            )
        audio_value = row_value(label_row, ("audio_path", "audio_file"))
        audio_path = resolve_audio_path(ROOT, audio_dir, audio_value)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for clip_id={label_row.clip_id}: {audio_path}")

        segment = load_audio_segment(
            audio_path,
            sample_rate=sample_rate,
            start_time=float(label_row.start_time),
            end_time=float(label_row.end_time),
        )
        windows = iter_windows(
            segment,
            sample_rate=sample_rate,
            window_seconds=float(audio_config["window_seconds"]),
            hop_seconds=float(audio_config["hop_seconds"]),
        )
        for window_index, (start_sample, end_sample, chunk) in enumerate(windows):
            feature = log_mel_feature(chunk, audio_config)
            feature_name = f"{safe_name(label_row.clip_id)}_{window_index:04d}.npz"
            feature_path = feature_dir / feature_name
            np.savez_compressed(feature_path, feature=feature)
            rows.append(
                {
                    "clip_id": label_row.clip_id,
                    "session_id": label_row.session_id,
                    "person_id": label_row.person_id,
                    "label": label,
                    "task_label": label,
                    "label_id": label_map[label],
                    "audio_file": str(audio_value),
                    "audio_path": str(audio_value),
                    "window_index": window_index,
                    "window_start": float(label_row.start_time) + start_sample / sample_rate,
                    "window_end": float(label_row.start_time) + end_sample / sample_rate,
                    "feature_path": str(Path("features") / feature_name),
                }
            )
            for column in (
                "task",
                "source_dataset",
                "quality",
                "split",
                "split_group",
                "hard_negative",
                "sample_weight",
                "negative_role",
                "source_audio_id",
                "notes",
            ):
                value = row_value(label_row, (column,), None)
                if value is not None:
                    rows[-1][column] = value

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    print(f"OK: wrote {len(manifest)} feature windows to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
