from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import iter_windows, load_audio_segment, log_mel_feature
from src.audio.models import build_small_cnn


ALLOWED_LABELS = {"cough", "non_cough"}
ALLOWED_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {
    "clip_id",
    "session_id",
    "person_id",
    "start_time",
    "end_time",
    "label",
    "audio_file",
    "split",
}
TRACE_COLUMNS = {"review_file", "review_id"}

BOARD_HTK_NO_NORM_V1 = {
    "sample_rate": 16000,
    "window_seconds": 1.0,
    "hop_seconds": 0.5,
    "n_mels": 40,
    "n_fft": 1024,
    "hop_length": 160,
    "fmin": 50,
    "fmax": 7600,
    "mel_scale": "htk",
    "htk": True,
    "center": True,
    "top_db": 80,
    "normalize": "none",
}
EXPECTED_FEATURE_SHAPE = (40, 101)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset, preprocessing, and one-epoch smoke checks for audio_labels_v3_candidate.csv."
    )
    parser.add_argument("--labels", default="data/labels/audio_labels_v3_candidate.csv")
    parser.add_argument("--dataset-out", default="results/check_audio_labels_v3_candidate")
    parser.add_argument("--preprocess-out", default="results/preprocess_smoke_v3_candidate")
    parser.add_argument("--training-out", default="results/one_epoch_smoke_v3_candidate")
    parser.add_argument("--preprocess-per-bucket", type=int, default=8)
    parser.add_argument("--train-per-bucket", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def resolve_audio_path(audio_value: object) -> Path:
    value = Path(str(audio_value))
    if value.is_absolute():
        return value
    root_relative = ROOT / value
    if root_relative.exists():
        return root_relative
    return value


def audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype(str).value_counts(dropna=False).sort_index().items()
    }


def split_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    if not {"split", "label"}.issubset(frame.columns):
        return {}
    pivot = (
        frame.groupby(["split", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted(ALLOWED_SPLITS), fill_value=0)
    )
    for label in sorted(ALLOWED_LABELS):
        if label not in pivot.columns:
            pivot[label] = 0
    return {
        str(split): {str(label): int(pivot.loc[split, label]) for label in sorted(ALLOWED_LABELS)}
        for split in pivot.index
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in values) + " |")
    return "\n".join(lines) + "\n"


def run_dataset_check(labels: pd.DataFrame, labels_path: Path, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(labels.columns))
    missing_trace_columns = sorted(TRACE_COLUMNS - set(labels.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    if missing_trace_columns:
        errors.append(f"Missing traceability columns: {missing_trace_columns}")

    report: dict[str, Any] = {
        "status": "failed",
        "labels_csv": str(labels_path),
        "rows": int(len(labels)),
        "errors": errors,
        "warnings": warnings,
        "missing_columns": missing_columns,
        "missing_trace_columns": missing_trace_columns,
    }

    if missing_columns:
        write_json(report, out_dir / "report.json")
        write_dataset_md(report, out_dir / "report.md")
        return report

    starts = to_number(labels["start_time"])
    ends = to_number(labels["end_time"])
    invalid_start_end = labels[starts.isna() | ends.isna()]
    bad_order = labels[ends <= starts]
    negative_start = labels[starts < 0]
    if not invalid_start_end.empty:
        errors.append(f"Rows with non-numeric start_time/end_time: {len(invalid_start_end)}")
    if not bad_order.empty:
        errors.append(f"Rows with end_time <= start_time: {len(bad_order)}")
    if not negative_start.empty:
        errors.append(f"Rows with start_time < 0: {len(negative_start)}")

    invalid_labels = sorted(set(labels["label"].astype(str)) - ALLOWED_LABELS)
    invalid_splits = sorted(set(labels["split"].astype(str)) - ALLOWED_SPLITS)
    if invalid_labels:
        errors.append(f"Invalid label values: {invalid_labels}")
    if invalid_splits:
        errors.append(f"Invalid split values: {invalid_splits}")

    clip_duplicates = labels[labels["clip_id"].astype(str).duplicated(keep=False)]
    if not clip_duplicates.empty:
        errors.append(f"Duplicate clip_id rows: {len(clip_duplicates)}")

    duplicate_key = labels.assign(
        _audio_file=labels["audio_file"].astype(str),
        _start=starts.round(6),
        _end=ends.round(6),
    )[["_audio_file", "_start", "_end"]]
    duplicate_audio_segments = labels[duplicate_key.duplicated(keep=False)]
    if not duplicate_audio_segments.empty:
        errors.append(f"Duplicate audio_file + start_time + end_time rows: {len(duplicate_audio_segments)}")

    leaking_people: dict[str, list[str]] = {}
    for person_id, group in labels.groupby(labels["person_id"].astype(str), dropna=False):
        splits = sorted(set(group["split"].astype(str)) - {""})
        if len(splits) > 1:
            leaking_people[str(person_id)] = splits
    if leaking_people:
        errors.append(f"person_id appears in multiple splits: {len(leaking_people)} people")

    audio_missing_rows: list[dict[str, Any]] = []
    duration_overrun_rows: list[dict[str, Any]] = []
    duration_cache: dict[str, float | None] = {}
    for audio_file in sorted(set(labels["audio_file"].astype(str))):
        audio_path = resolve_audio_path(audio_file)
        if not audio_path.exists():
            duration_cache[audio_file] = None
            continue
        duration_cache[audio_file] = audio_duration_seconds(audio_path)

    for index, row in labels.iterrows():
        audio_file = str(row["audio_file"])
        audio_path = resolve_audio_path(audio_file)
        if not audio_path.exists():
            audio_missing_rows.append({"row": int(index), "clip_id": row["clip_id"], "audio_file": audio_file})
            continue
        duration = duration_cache.get(audio_file)
        if duration is not None and not pd.isna(ends.iloc[index]) and float(ends.iloc[index]) > duration + 1e-6:
            duration_overrun_rows.append(
                {
                    "row": int(index),
                    "clip_id": row["clip_id"],
                    "audio_file": audio_file,
                    "end_time": float(ends.iloc[index]),
                    "duration": round(float(duration), 6),
                }
            )
    if audio_missing_rows:
        errors.append(f"Missing audio_file rows: {len(audio_missing_rows)}")
    if duration_overrun_rows:
        errors.append(f"Rows where end_time exceeds WAV duration: {len(duration_overrun_rows)}")

    sample_weight = to_number(labels.get("sample_weight", pd.Series([], dtype=float)))
    missing_sample_weight = int(labels.get("sample_weight", pd.Series([""] * len(labels))).astype(str).str.strip().eq("").sum())
    non_positive_sample_weight = int((sample_weight.notna() & (sample_weight <= 0)).sum())
    if missing_sample_weight:
        warnings.append(f"sample_weight is empty on {missing_sample_weight} inherited/base rows")
    if non_positive_sample_weight:
        errors.append(f"sample_weight <= 0 rows: {non_positive_sample_weight}")

    generated_rows = labels["clip_id"].astype(str).str.startswith("yamnet_review_")
    trace_missing: dict[str, int] = {}
    for column in TRACE_COLUMNS:
        if column in labels.columns:
            trace_missing[column] = int(labels.loc[generated_rows, column].astype(str).str.strip().eq("").sum())
    if any(count > 0 for count in trace_missing.values()):
        errors.append(f"Generated rows missing traceability fields: {trace_missing}")

    hard_negative_flags = labels.get("hard_negative", pd.Series([""] * len(labels))).map(parse_bool)
    source_audio = labels.get("source_audio", pd.Series([""] * len(labels))).astype(str)
    source_audio_non_empty = source_audio[source_audio.str.strip().ne("")]
    source_audio_counts = source_audio_non_empty.value_counts()

    report.update(
        {
            "status": "passed" if not errors else "failed",
            "label_counts": value_counts(labels, "label"),
            "split_counts": value_counts(labels, "split"),
            "split_label_counts": split_label_counts(labels),
            "hard_negative_count": int(hard_negative_flags.sum()),
            "hard_negative_source_counts": value_counts(labels, "hard_negative_source"),
            "source_audio": {
                "non_empty_rows": int(source_audio_non_empty.shape[0]),
                "unique_non_empty": int(source_audio_non_empty.nunique()),
                "max_samples_per_source_audio": int(source_audio_counts.max()) if not source_audio_counts.empty else 0,
                "top_source_audio_counts": {
                    str(key): int(value) for key, value in source_audio_counts.head(20).items()
                },
            },
            "sample_weight": {
                "missing_rows": missing_sample_weight,
                "numeric_positive_rows": int((sample_weight > 0).sum()),
                "non_positive_rows": non_positive_sample_weight,
            },
            "traceability": {
                "required_columns_present": not missing_trace_columns,
                "generated_row_missing_counts": trace_missing,
            },
            "audio_files": {
                "unique": int(labels["audio_file"].astype(str).nunique()),
                "missing_rows": len(audio_missing_rows),
                "duration_overrun_rows": len(duration_overrun_rows),
                "missing_examples": audio_missing_rows[:20],
                "duration_overrun_examples": duration_overrun_rows[:20],
            },
            "clip_id_duplicate_rows": int(len(clip_duplicates)),
            "duplicate_audio_segment_rows": int(len(duplicate_audio_segments)),
            "person_id_leakage_count": int(len(leaking_people)),
            "person_id_leakage_examples": dict(list(leaking_people.items())[:20]),
        }
    )
    write_json(report, out_dir / "report.json")
    write_dataset_md(report, out_dir / "report.md")
    return report


def write_dataset_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# audio_labels_v3_candidate Dataset Check",
        "",
        f"- status: {report.get('status')}",
        f"- rows: {report.get('rows')}",
        f"- labels_csv: `{report.get('labels_csv')}`",
        "",
        "## Counts",
        "",
        f"- label_counts: `{report.get('label_counts', {})}`",
        f"- split_counts: `{report.get('split_counts', {})}`",
        f"- hard_negative_count: `{report.get('hard_negative_count')}`",
        f"- hard_negative_source_counts: `{report.get('hard_negative_source_counts', {})}`",
        "",
        "## Split x Label",
        "",
    ]
    split_rows = []
    for split, counts in report.get("split_label_counts", {}).items():
        row = {"split": split}
        row.update(counts)
        split_rows.append(row)
    lines.append(markdown_table(split_rows, ["split", "cough", "non_cough"]))
    lines.extend(
        [
            "## Source Audio",
            "",
            f"- non_empty_rows: `{report.get('source_audio', {}).get('non_empty_rows')}`",
            f"- unique_non_empty: `{report.get('source_audio', {}).get('unique_non_empty')}`",
            f"- max_samples_per_source_audio: `{report.get('source_audio', {}).get('max_samples_per_source_audio')}`",
            "",
            "## Validation",
            "",
            f"- sample_weight: `{report.get('sample_weight', {})}`",
            f"- traceability: `{report.get('traceability', {})}`",
            f"- audio_files: `{report.get('audio_files', {})}`",
            f"- person_id_leakage_count: `{report.get('person_id_leakage_count')}`",
            f"- duplicate_audio_segment_rows: `{report.get('duplicate_audio_segment_rows')}`",
            "",
            "## Errors",
            "",
        ]
    )
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] if errors else ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bucket_masks(labels: pd.DataFrame) -> dict[str, pd.Series]:
    hard_negative = labels.get("hard_negative", pd.Series([""] * len(labels))).map(parse_bool)
    hard_source = labels.get("hard_negative_source", pd.Series([""] * len(labels))).astype(str)
    label = labels["label"].astype(str)
    return {
        "cough": label.eq("cough"),
        "clean_non_cough": label.eq("non_cough") & ~hard_negative,
        "model_false_alarm_hard_negative": label.eq("non_cough") & hard_negative & hard_source.eq("model_false_alarm"),
        "human_cough_like_hard_negative": label.eq("non_cough") & hard_negative & hard_source.eq("human_cough_like"),
        "hard_negative": label.eq("non_cough") & hard_negative,
    }


def sample_bucket(labels: pd.DataFrame, mask: pd.Series, count: int, seed: int) -> pd.DataFrame:
    candidates = labels[mask].copy()
    if candidates.empty:
        return candidates
    sample_count = min(int(count), len(candidates))
    return candidates.sample(n=sample_count, random_state=seed).copy()


def extract_features_for_row(row: pd.Series, audio_config: dict[str, Any], max_windows: int | None = None) -> list[np.ndarray]:
    audio_path = resolve_audio_path(row["audio_file"])
    segment = load_audio_segment(
        audio_path,
        sample_rate=int(audio_config["sample_rate"]),
        start_time=float(row["start_time"]),
        end_time=float(row["end_time"]),
    )
    windows = iter_windows(
        segment,
        sample_rate=int(audio_config["sample_rate"]),
        window_seconds=float(audio_config["window_seconds"]),
        hop_seconds=float(audio_config["hop_seconds"]),
    )
    features: list[np.ndarray] = []
    for _, _, chunk in windows:
        features.append(log_mel_feature(chunk, audio_config))
        if max_windows is not None and len(features) >= max_windows:
            break
    return features


def feature_stats(features: list[np.ndarray]) -> dict[str, float | int | list[int]]:
    if not features:
        return {"count": 0}
    stacked = np.stack(features)
    return {
        "count": int(stacked.shape[0]),
        "shape": [int(value) for value in features[0].shape],
        "mean": float(stacked.mean()),
        "std": float(stacked.std()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
    }


def run_preprocess_smoke(labels: pd.DataFrame, out_dir: Path, per_bucket: int, seed: int) -> dict[str, Any]:
    ensure_dir(out_dir)
    masks = bucket_masks(labels)
    selected_frames = []
    for index, bucket in enumerate(
        [
            "cough",
            "clean_non_cough",
            "model_false_alarm_hard_negative",
            "human_cough_like_hard_negative",
        ]
    ):
        frame = sample_bucket(labels, masks[bucket], per_bucket, seed + index)
        frame["preprocess_bucket"] = bucket
        selected_frames.append(frame)
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()

    errors: list[str] = []
    warnings: list[str] = []
    all_features: list[np.ndarray] = []
    bucket_feature_counts: Counter[str] = Counter()
    nan_inf_count = 0
    bad_shape_count = 0
    checked_rows: list[dict[str, Any]] = []

    for row in selected.to_dict("records"):
        try:
            features = extract_features_for_row(pd.Series(row), BOARD_HTK_NO_NORM_V1)
        except Exception as exc:
            errors.append(f"feature extraction failed clip_id={row.get('clip_id')}: {exc}")
            continue
        if not features:
            errors.append(f"no windows produced clip_id={row.get('clip_id')}")
            continue
        for feature in features:
            if tuple(feature.shape) != EXPECTED_FEATURE_SHAPE:
                bad_shape_count += 1
            if not np.isfinite(feature).all():
                nan_inf_count += 1
            all_features.append(feature)
            bucket_feature_counts[str(row["preprocess_bucket"])] += 1
        checked_rows.append(
            {
                "clip_id": row.get("clip_id"),
                "bucket": row.get("preprocess_bucket"),
                "windows": len(features),
                "feature_shape": list(features[0].shape),
            }
        )

    if bad_shape_count:
        errors.append(f"features with shape != {EXPECTED_FEATURE_SHAPE}: {bad_shape_count}")
    if nan_inf_count:
        errors.append(f"features containing NaN/Inf: {nan_inf_count}")

    selected_counts = selected["preprocess_bucket"].value_counts().to_dict() if not selected.empty else {}
    for bucket in [
        "cough",
        "clean_non_cough",
        "model_false_alarm_hard_negative",
        "human_cough_like_hard_negative",
    ]:
        if int(selected_counts.get(bucket, 0)) == 0:
            warnings.append(f"No selected rows for preprocess bucket: {bucket}")

    report: dict[str, Any] = {
        "status": "passed" if not errors else "failed",
        "audio_frontend": "board_htk_no_norm_v1",
        "audio_config": BOARD_HTK_NO_NORM_V1,
        "expected_feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "selected_rows": int(len(selected)),
        "selected_row_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "feature_counts_by_bucket": {str(k): int(v) for k, v in bucket_feature_counts.items()},
        "feature_stats": feature_stats(all_features),
        "nan_inf_feature_count": int(nan_inf_count),
        "bad_shape_feature_count": int(bad_shape_count),
        "checked_rows_preview": checked_rows[:20],
        "errors": errors,
        "warnings": warnings,
    }
    write_json(report, out_dir / "report.json")
    write_preprocess_md(report, out_dir / "report.md")
    return report


def write_preprocess_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3 Candidate Preprocess Smoke Test",
        "",
        f"- status: {report['status']}",
        f"- audio_frontend: `{report['audio_frontend']}`",
        f"- selected_rows: {report['selected_rows']}",
        f"- expected_feature_shape: `{report['expected_feature_shape']}`",
        f"- nan_inf_feature_count: {report['nan_inf_feature_count']}",
        f"- bad_shape_feature_count: {report['bad_shape_feature_count']}",
        "",
        "## Counts",
        "",
        f"- selected_row_counts: `{report['selected_row_counts']}`",
        f"- feature_counts_by_bucket: `{report['feature_counts_by_bucket']}`",
        "",
        "## Feature Stats",
        "",
        f"`{report['feature_stats']}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {error}" for error in report["errors"]] if report["errors"] else ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in report["warnings"]] if report["warnings"] else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def row_category(row: pd.Series) -> str:
    if str(row["label"]) == "cough":
        return "cough"
    hard_negative = parse_bool(row.get("hard_negative", ""))
    if hard_negative:
        return "hard_negative"
    return "clean_non_cough"


def build_training_subset(labels: pd.DataFrame, per_bucket: int, seed: int) -> pd.DataFrame:
    masks = bucket_masks(labels)
    frames = [
        sample_bucket(labels, masks["cough"], per_bucket, seed + 10),
        sample_bucket(labels, masks["clean_non_cough"], per_bucket, seed + 11),
        sample_bucket(labels, masks["hard_negative"], per_bucket, seed + 12),
    ]
    subset = pd.concat(frames, ignore_index=True)
    subset["smoke_category"] = subset.apply(row_category, axis=1)
    subset = subset.drop_duplicates(subset=["audio_file", "start_time", "end_time"], keep="first")
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def assign_smoke_splits(subset: pd.DataFrame, seed: int) -> pd.Series:
    rng = random.Random(seed)
    split_by_index = pd.Series(index=subset.index, dtype=object)
    used_groups: dict[str, str] = {}
    for category in ["cough", "clean_non_cough", "hard_negative"]:
        indices = subset.index[subset["smoke_category"] == category].tolist()
        groups: dict[str, list[int]] = {}
        for index in indices:
            groups.setdefault(str(subset.loc[index, "person_id"]), []).append(index)
        group_ids = list(groups)
        rng.shuffle(group_ids)
        total = len(group_ids)
        if total == 0:
            continue
        train_n = max(1, int(math.floor(total * 0.7)))
        val_n = max(1, int(math.floor(total * 0.15))) if total >= 3 else 0
        if train_n + val_n >= total and total >= 3:
            train_n = max(1, total - 2)
            val_n = 1
        for pos, group_id in enumerate(group_ids):
            if group_id in used_groups:
                split = used_groups[group_id]
            elif pos < train_n:
                split = "train"
            elif pos < train_n + val_n:
                split = "val"
            else:
                split = "test"
            used_groups[group_id] = split
            for index in groups[group_id]:
                split_by_index.loc[index] = split
    split_by_index = split_by_index.fillna("train")
    return split_by_index


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    total = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "rows": len(y_true),
    }


def run_one_epoch_smoke(labels: pd.DataFrame, out_dir: Path, per_bucket: int, seed: int, batch_size: int) -> dict[str, Any]:
    ensure_dir(out_dir)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("torch is required for one-epoch training smoke test") from exc

    torch.manual_seed(seed)
    subset = build_training_subset(labels, per_bucket, seed)
    subset["smoke_split"] = assign_smoke_splits(subset, seed)

    errors: list[str] = []
    warnings: list[str] = []
    features: list[np.ndarray] = []
    labels_numeric: list[int] = []
    splits: list[str] = []
    categories: list[str] = []
    used_rows: list[dict[str, Any]] = []

    for row in subset.to_dict("records"):
        try:
            extracted = extract_features_for_row(pd.Series(row), BOARD_HTK_NO_NORM_V1, max_windows=1)
        except Exception as exc:
            errors.append(f"training feature extraction failed clip_id={row.get('clip_id')}: {exc}")
            continue
        if not extracted:
            errors.append(f"training row produced no windows clip_id={row.get('clip_id')}")
            continue
        feature = extracted[0]
        if tuple(feature.shape) != EXPECTED_FEATURE_SHAPE or not np.isfinite(feature).all():
            errors.append(f"invalid training feature clip_id={row.get('clip_id')} shape={feature.shape}")
            continue
        features.append(feature)
        labels_numeric.append(1 if str(row["label"]) == "cough" else 0)
        splits.append(str(row["smoke_split"]))
        categories.append(str(row["smoke_category"]))
        used_rows.append(
            {
                "clip_id": row.get("clip_id"),
                "label": row.get("label"),
                "category": row.get("smoke_category"),
                "split": row.get("smoke_split"),
                "person_id": row.get("person_id"),
            }
        )

    if errors:
        report = {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "selected_rows": int(len(subset)),
            "used_rows": len(used_rows),
        }
        write_json(report, out_dir / "metrics.json")
        write_training_md(report, out_dir / "report.md")
        return report

    split_counts = Counter(splits)
    if set(split_counts) != {"train", "val", "test"}:
        warnings.append(f"Smoke splits are not complete: {dict(split_counts)}")
    category_counts = Counter(categories)
    split_category_counts = (
        pd.DataFrame({"split": splits, "category": categories}).groupby(["split", "category"]).size().to_dict()
    )

    tensor_x = torch.tensor(np.stack(features), dtype=torch.float32).unsqueeze(1)
    tensor_y = torch.tensor(labels_numeric, dtype=torch.long)
    split_array = np.array(splits)

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        indices = np.where(split_array == split)[0]
        if indices.size == 0:
            raise ValueError(f"No rows in smoke split: {split}")
        dataset = TensorDataset(tensor_x[indices], tensor_y[indices])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader("train", True)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_small_cnn(num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    model.train()
    train_loss_total = 0.0
    train_count = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        train_loss_total += float(loss.item()) * int(batch_y.size(0))
        train_count += int(batch_y.size(0))
    train_loss = train_loss_total / max(train_count, 1)

    def predict_loader(loader: DataLoader) -> tuple[list[int], list[int]]:
        y_true: list[int] = []
        y_pred: list[int] = []
        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                logits = model(batch_x.to(device))
                pred = logits.argmax(dim=1).cpu().tolist()
                y_pred.extend(int(value) for value in pred)
                y_true.extend(int(value) for value in batch_y.tolist())
        return y_true, y_pred

    val_true, val_pred = predict_loader(val_loader)
    test_true, test_pred = predict_loader(test_loader)
    report = {
        "status": "passed",
        "purpose": "one_epoch_training_smoke_test_only_not_formal_training",
        "model_saved": False,
        "audio_frontend": "board_htk_no_norm_v1",
        "device": str(device),
        "epochs_run": 1,
        "train_loss": train_loss,
        "selected_rows": int(len(subset)),
        "used_feature_rows": int(len(features)),
        "category_counts": {str(k): int(v) for k, v in category_counts.items()},
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "split_category_counts": {f"{key[0]}:{key[1]}": int(value) for key, value in split_category_counts.items()},
        "feature_shape": list(EXPECTED_FEATURE_SHAPE),
        "val": compute_metrics(val_true, val_pred),
        "test": compute_metrics(test_true, test_pred),
        "used_rows_preview": used_rows[:30],
        "errors": errors,
        "warnings": warnings,
    }
    write_json(report, out_dir / "metrics.json")
    pd.DataFrame(used_rows).to_csv(out_dir / "smoke_subset.csv", index=False, encoding="utf-8")
    pd.DataFrame([{"epoch": 1, "train_loss": train_loss}]).to_csv(out_dir / "train_log.csv", index=False)
    write_training_md(report, out_dir / "report.md")
    return report


def write_training_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3 Candidate One-Epoch Training Smoke Test",
        "",
        f"- status: {report.get('status')}",
        f"- purpose: `{report.get('purpose', 'one_epoch_training_smoke_test_only_not_formal_training')}`",
        f"- model_saved: `{report.get('model_saved', False)}`",
        f"- epochs_run: {report.get('epochs_run')}",
        f"- selected_rows: {report.get('selected_rows')}",
        f"- used_feature_rows: {report.get('used_feature_rows')}",
        f"- train_loss: {report.get('train_loss')}",
        "",
        "## Counts",
        "",
        f"- category_counts: `{report.get('category_counts', {})}`",
        f"- split_counts: `{report.get('split_counts', {})}`",
        f"- split_category_counts: `{report.get('split_category_counts', {})}`",
        "",
        "## Minimal Evaluation",
        "",
        f"- val: `{report.get('val', {})}`",
        f"- test: `{report.get('test', {})}`",
        "",
        "## Errors",
        "",
    ]
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] if errors else ["- none"])
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels_path = resolve_path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    labels = read_labels(labels_path)
    dataset_report = run_dataset_check(labels, labels_path, resolve_path(args.dataset_out))
    preprocess_report = run_preprocess_smoke(
        labels,
        resolve_path(args.preprocess_out),
        per_bucket=args.preprocess_per_bucket,
        seed=args.seed,
    )
    training_report = run_one_epoch_smoke(
        labels,
        resolve_path(args.training_out),
        per_bucket=args.train_per_bucket,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    overall_passed = all(
        report.get("status") == "passed"
        for report in [dataset_report, preprocess_report, training_report]
    )
    summary = {
        "status": "passed" if overall_passed else "failed",
        "dataset_check": dataset_report.get("status"),
        "preprocess_smoke": preprocess_report.get("status"),
        "one_epoch_training_smoke": training_report.get("status"),
        "outputs": {
            "dataset_report_md": str(resolve_path(args.dataset_out) / "report.md"),
            "dataset_report_json": str(resolve_path(args.dataset_out) / "report.json"),
            "preprocess_report_md": str(resolve_path(args.preprocess_out) / "report.md"),
            "preprocess_report_json": str(resolve_path(args.preprocess_out) / "report.json"),
            "training_report_md": str(resolve_path(args.training_out) / "report.md"),
            "training_metrics_json": str(resolve_path(args.training_out) / "metrics.json"),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
