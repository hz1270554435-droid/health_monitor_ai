from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CANONICAL_COLUMNS = [
    "clip_id",
    "session_id",
    "person_id",
    "source_dataset",
    "audio_path",
    "start_time",
    "end_time",
    "original_label",
    "normalized_label",
    "task",
    "task_label",
    "quality",
    "split_group",
    "notes",
    "split",
    "hard_negative",
    "sample_weight",
    "negative_role",
    "source_audio_id",
    "eval_allowed",
]

REQUIRED_CANONICAL_COLUMNS = {
    "clip_id",
    "session_id",
    "person_id",
    "source_dataset",
    "audio_path",
    "start_time",
    "end_time",
    "original_label",
    "normalized_label",
    "quality",
    "split_group",
    "source_audio_id",
}


def resolve_path(path: str | Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def display_path(path: str | Path, root: Path = ROOT) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(value)


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def bool_value(value: object, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def first_present(row: pd.Series, names: tuple[str, ...], default: object = "") -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def resolve_audio_reference(audio_root: str | Path, audio_value: object, root: Path = ROOT) -> Path:
    audio_path = Path(str(audio_value))
    if audio_path.is_absolute():
        return audio_path
    base = resolve_path(audio_root, root)
    rooted = base / audio_path
    root_relative = root / audio_path
    if not rooted.exists() and root_relative.exists():
        return root_relative
    return rooted


def canonicalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output = output[CANONICAL_COLUMNS]
    output["start_time"] = pd.to_numeric(output["start_time"], errors="coerce")
    output["end_time"] = pd.to_numeric(output["end_time"], errors="coerce")
    output["hard_negative"] = output["hard_negative"].map(lambda value: bool_value(value, False))
    output["eval_allowed"] = output["eval_allowed"].map(lambda value: bool_value(value, True))
    output["sample_weight"] = pd.to_numeric(output["sample_weight"], errors="coerce").fillna(1.0)
    for column in output.columns:
        if column not in {"start_time", "end_time", "hard_negative", "eval_allowed", "sample_weight"}:
            output[column] = output[column].fillna("").astype(str)
    return output


def rows_from_label_csv(source: dict[str, Any], root: Path = ROOT) -> list[dict[str, object]]:
    path = resolve_path(source["path"], root)
    if not path.exists():
        raise FileNotFoundError(f"Label source not found: {path}")
    frame = pd.read_csv(path)
    audio_root = source.get("audio_root", ".")
    source_dataset = str(source.get("source_dataset") or source.get("name") or path.stem)
    default_quality = str(source.get("quality", "reviewed"))
    default_eval_allowed = bool(source.get("eval_allowed", True))
    default_negative_role = str(source.get("negative_role", ""))
    rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        audio_value = first_present(row, ("audio_path", "audio_file"))
        if audio_value == "":
            raise ValueError(f"{path} row {index} is missing audio_path/audio_file")
        audio_path = resolve_audio_reference(audio_root, audio_value, root)
        original_label = str(first_present(row, ("original_label", "label", "task_label"), ""))
        normalized_label = normalize_label(first_present(row, ("normalized_label", "label", "task_label"), original_label))
        start_time = float(first_present(row, ("start_time",), 0.0))
        end_time = float(first_present(row, ("end_time",), 0.0))
        source_audio_id = str(
            first_present(row, ("source_audio_id",), stable_id("audio", display_path(audio_path, root)))
        )
        clip_id = str(
            first_present(
                row,
                ("clip_id",),
                stable_id("clip", source_dataset, display_path(audio_path, root), start_time, end_time, normalized_label),
            )
        )
        session_id = str(first_present(row, ("session_id",), f"session_{clip_id}"))
        person_id = str(first_present(row, ("person_id",), f"UNKNOWN_{source_audio_id}"))
        split_group = str(first_present(row, ("split_group",), person_id))
        rows.append(
            {
                "clip_id": clip_id,
                "session_id": session_id,
                "person_id": person_id,
                "source_dataset": source_dataset,
                "audio_path": display_path(audio_path, root),
                "start_time": start_time,
                "end_time": end_time,
                "original_label": original_label,
                "normalized_label": normalized_label,
                "task": str(first_present(row, ("task",), "")),
                "task_label": str(first_present(row, ("task_label",), "")),
                "quality": str(first_present(row, ("quality",), default_quality)),
                "split_group": split_group,
                "notes": str(first_present(row, ("notes",), "")),
                "split": str(first_present(row, ("split",), "")),
                "hard_negative": bool_value(first_present(row, ("hard_negative",), False)),
                "sample_weight": float(first_present(row, ("sample_weight",), 1.0) or 1.0),
                "negative_role": str(first_present(row, ("negative_role",), default_negative_role)),
                "source_audio_id": source_audio_id,
                "eval_allowed": bool_value(first_present(row, ("eval_allowed",), default_eval_allowed), default_eval_allowed),
            }
        )
    return rows


def rows_from_hard_data_manifest(source: dict[str, Any], root: Path = ROOT) -> list[dict[str, object]]:
    path = resolve_path(source["path"], root)
    if not path.exists():
        raise FileNotFoundError(f"Hard-data manifest not found: {path}")
    frame = pd.read_csv(path)
    source_dataset = str(source.get("source_dataset") or source.get("name") or path.stem)
    label = normalize_label(source.get("label", "non_cough"))
    quality = str(source.get("quality", "external_pool"))
    eval_allowed = bool(source.get("eval_allowed", False))
    rows: list[dict[str, object]] = []

    for index, row in frame.iterrows():
        audio_value = first_present(row, ("copied_audio", "audio_path", "audio_file", "source_audio"))
        if audio_value == "":
            raise ValueError(f"{path} row {index} is missing copied_audio/audio_path/source_audio")
        audio_path = resolve_path(str(audio_value), root)
        source_audio = str(first_present(row, ("source_audio",), audio_value))
        duration = float(first_present(row, ("duration",), 0.0))
        source_audio_id = stable_id("audio", source_audio)
        clip_id = stable_id("pool", source_dataset, source_audio, 0.0, duration, label)
        person_id = str(first_present(row, ("person_id",), f"UNKNOWN_{source_audio_id}"))
        session_id = str(first_present(row, ("session_id",), f"session_{clip_id}"))
        rows.append(
            {
                "clip_id": clip_id,
                "session_id": session_id,
                "person_id": person_id,
                "source_dataset": source_dataset,
                "audio_path": display_path(audio_path, root),
                "start_time": 0.0,
                "end_time": duration,
                "original_label": label,
                "normalized_label": label,
                "task": "",
                "task_label": "",
                "quality": quality,
                "split_group": person_id,
                "notes": str(first_present(row, ("notes",), "")),
                "split": "train" if not eval_allowed else str(first_present(row, ("split",), "")),
                "hard_negative": bool(source.get("hard_negative", False)),
                "sample_weight": float(source.get("sample_weight", 1.0)),
                "negative_role": str(source.get("negative_role", "shared_negative")),
                "source_audio_id": source_audio_id,
                "eval_allowed": eval_allowed,
            }
        )
    return rows


def build_audio_manifest(config: dict[str, Any], root: Path = ROOT) -> pd.DataFrame:
    manifest_cfg = config.get("manifest", {})
    sources = manifest_cfg.get("sources", config.get("sources", {}))
    rows: list[dict[str, object]] = []
    for source in sources.get("label_csvs", []):
        rows.extend(rows_from_label_csv(source, root=root))
    for source in sources.get("hard_data_manifests", []):
        rows.extend(rows_from_hard_data_manifest(source, root=root))
    manifest = canonicalize_manifest(pd.DataFrame(rows))
    validate_manifest(manifest)
    return manifest


def validate_manifest(frame: pd.DataFrame, require_files: bool = False, root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_CANONICAL_COLUMNS - set(frame.columns))
    if missing:
        errors.append(f"Manifest missing columns: {missing}")
        if errors:
            raise ValueError("; ".join(errors))

    duplicated = frame[frame["clip_id"].duplicated()]["clip_id"].astype(str).tolist()
    if duplicated:
        errors.append(f"Duplicate clip_id values: {duplicated[:20]}")

    starts = pd.to_numeric(frame["start_time"], errors="coerce")
    ends = pd.to_numeric(frame["end_time"], errors="coerce")
    if starts.isna().any() or ends.isna().any():
        errors.append("start_time/end_time must be numeric")
    if (starts < 0).any():
        errors.append("start_time contains negative values")
    if (ends <= starts).any():
        errors.append("end_time must be greater than start_time")

    if require_files:
        missing_files = [
            str(resolve_path(value, root))
            for value in frame["audio_path"].astype(str)
            if not resolve_path(value, root).exists()
        ]
        if missing_files:
            errors.append(f"Missing audio files: {missing_files[:20]}")

    if errors:
        raise ValueError("; ".join(errors))
    return errors


def write_manifest(frame: pd.DataFrame, path: str | Path, root: Path = ROOT) -> Path:
    from src.common.io import ensure_dir

    target = resolve_path(path, root)
    ensure_dir(target.parent)
    frame.to_csv(target, index=False)
    return target
