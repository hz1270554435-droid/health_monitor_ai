from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.audio.manifest import CANONICAL_COLUMNS, canonicalize_manifest, normalize_label, resolve_path
from src.common.splits import assign_group_splits


TASK_VIEW_COLUMNS = [
    *CANONICAL_COLUMNS,
    "label",
    "audio_file",
    "label_id",
]


def load_csv(path: str | Path, root: Path) -> pd.DataFrame:
    target = resolve_path(path, root)
    if not target.exists():
        raise FileNotFoundError(f"CSV not found: {target}")
    return pd.read_csv(target)


def build_non_cough_pool(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pool_cfg = config.get("non_cough_pool", config.get("pool", {}))
    labels = {normalize_label(value) for value in pool_cfg.get("normalized_labels", ["non_cough"])}
    output = canonicalize_manifest(manifest)
    output = output[output["normalized_label"].map(normalize_label).isin(labels)].copy()
    output["negative_role"] = output["negative_role"].where(
        output["negative_role"].astype(str).str.len() > 0,
        output["hard_negative"].map(lambda value: "hard_negative" if bool(value) else "shared_negative"),
    )
    output["task"] = ""
    output["task_label"] = ""
    output = output.drop_duplicates(subset=["clip_id"], keep="first").reset_index(drop=True)
    return output


def split_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get(
        "split",
        {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "group_column": "split_group",
        },
    )


def assign_task_splits(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    cfg = split_config(config)
    group_column = str(cfg.get("group_column", "split_group"))
    if group_column not in output.columns:
        if "split_group" in output.columns:
            group_column = "split_group"
        elif "person_id" in output.columns:
            group_column = "person_id"
        else:
            raise ValueError("Task view has no split_group/person_id column for grouped split")

    output["split"] = output.get("split", "").fillna("").astype(str)
    output.loc[output["split"].str.lower().isin({"nan", "none"}), "split"] = ""
    if "eval_allowed" in output.columns:
        train_only = ~output["eval_allowed"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        output.loc[train_only & output["split"].eq(""), "split"] = "train"

    group_existing = (
        output[output["split"].ne("")]
        .groupby(group_column)["split"]
        .agg(lambda values: sorted(set(str(value) for value in values if str(value))))
    )
    for group, splits in group_existing.items():
        if len(splits) == 1:
            output.loc[output[group_column].astype(str).eq(str(group)) & output["split"].eq(""), "split"] = splits[0]

    missing = output["split"].eq("")
    if missing.any():
        missing_groups = output.loc[missing, group_column].astype(str).nunique()
        allow_train_only = bool(cfg.get("allow_train_only_when_too_few_groups", True))
        if missing_groups >= 3:
            assigned = assign_group_splits(
                output.loc[missing].copy(),
                group_column=group_column,
                train_ratio=float(cfg.get("train_ratio", 0.7)),
                val_ratio=float(cfg.get("val_ratio", 0.15)),
                test_ratio=float(cfg.get("test_ratio", 0.15)),
                seed=int(config.get("project", {}).get("seed", 42)),
            )
            output.loc[missing, "split"] = assigned["split"].to_list()
        elif allow_train_only:
            output.loc[missing, "split"] = "train"
        else:
            raise ValueError(
                f"Need at least 3 unique {group_column} values to split missing rows; got {missing_groups}"
            )

    leaking = output.groupby(group_column)["split"].nunique(dropna=True)
    leaking = leaking[leaking > 1]
    if not leaking.empty:
        raise ValueError(f"{group_column} appears in multiple splits: {sorted(leaking.index.astype(str))}")
    return output


def _negative_rows(
    manifest: pd.DataFrame,
    non_cough_pool: pd.DataFrame | None,
    negative_labels: set[str],
    include_pool: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if include_pool and non_cough_pool is not None and not non_cough_pool.empty:
        pool = canonicalize_manifest(non_cough_pool)
        frames.append(pool[pool["normalized_label"].map(normalize_label).isin(negative_labels)].copy())

    manifest_negatives = manifest[manifest["normalized_label"].map(normalize_label).isin(negative_labels)].copy()
    if frames:
        pool_ids = set(pd.concat(frames, ignore_index=True)["clip_id"].astype(str))
        manifest_negatives = manifest_negatives[~manifest_negatives["clip_id"].astype(str).isin(pool_ids)]
    frames.append(manifest_negatives)
    return pd.concat(frames, ignore_index=True) if frames else manifest.iloc[0:0].copy()


def build_task_view(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    non_cough_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    task_cfg = config["task"]
    positive_label = normalize_label(task_cfg["positive_label"])
    negative_label = normalize_label(task_cfg["negative_label"])
    positive_source_labels = {
        normalize_label(value)
        for value in task_cfg.get("positive_source_labels", [positive_label])
    }
    negative_source_labels = {
        normalize_label(value)
        for value in task_cfg.get("negative_source_labels", ["non_cough"])
    }
    include_pool = bool(task_cfg.get("negative_pool", {}).get("include", True))
    task_name = str(task_cfg.get("name", config.get("run", {}).get("name", positive_label)))

    canonical = canonicalize_manifest(manifest)
    positives = canonical[canonical["normalized_label"].map(normalize_label).isin(positive_source_labels)].copy()
    negatives = _negative_rows(canonical, non_cough_pool, negative_source_labels, include_pool)

    positives["task_label"] = positive_label
    positives["label_id"] = 1
    positives["negative_role"] = ""
    negatives["task_label"] = negative_label
    negatives["label_id"] = 0
    negatives["negative_role"] = negatives["negative_role"].where(
        negatives["negative_role"].astype(str).str.len() > 0,
        negatives["hard_negative"].map(lambda value: "hard_negative" if bool(value) else "shared_negative"),
    )

    view = pd.concat([positives, negatives], ignore_index=True)
    if view.empty:
        raise ValueError(f"Task view {task_name} has no rows")
    view = view.drop_duplicates(subset=["clip_id"], keep="first").reset_index(drop=True)
    view["task"] = task_name
    view["label"] = view["task_label"]
    view["audio_file"] = view["audio_path"]
    view = assign_task_splits(view, config)

    for column in TASK_VIEW_COLUMNS:
        if column not in view.columns:
            view[column] = ""
    view = view[TASK_VIEW_COLUMNS]
    validate_task_view(view, config)
    return view


def validate_task_view(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    task_cfg = config["task"]
    positive_label = normalize_label(task_cfg["positive_label"])
    negative_label = normalize_label(task_cfg["negative_label"])
    required = {
        "clip_id",
        "session_id",
        "person_id",
        "audio_path",
        "start_time",
        "end_time",
        "task",
        "task_label",
        "label",
        "label_id",
        "split",
        "split_group",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"Task view missing columns: {missing}")
        raise ValueError("; ".join(errors))

    allowed_labels = {negative_label, positive_label}
    labels = {normalize_label(value) for value in frame["task_label"].astype(str)}
    invalid_labels = sorted(labels - allowed_labels)
    if invalid_labels:
        errors.append(f"Invalid task_label values: {invalid_labels}; allowed={sorted(allowed_labels)}")

    duplicated = frame[frame["clip_id"].duplicated()]["clip_id"].astype(str).tolist()
    if duplicated:
        errors.append(f"Duplicate clip_id values in task view: {duplicated[:20]}")

    expected_ids = {negative_label: 0, positive_label: 1}
    bad_ids = []
    for _, row in frame.iterrows():
        label = normalize_label(row["task_label"])
        if int(row["label_id"]) != expected_ids.get(label, -1):
            bad_ids.append(str(row["clip_id"]))
    if bad_ids:
        errors.append(f"label_id does not match task_label for clip_id values: {bad_ids[:20]}")

    split_values = set(frame["split"].dropna().astype(str))
    invalid_splits = sorted(split_values - {"train", "val", "test"})
    if invalid_splits:
        errors.append(f"Invalid split values: {invalid_splits}")

    group_column = str(split_config(config).get("group_column", "split_group"))
    if group_column in frame.columns:
        leaking = frame.groupby(group_column)["split"].nunique(dropna=True)
        leaking = leaking[leaking > 1]
        if not leaking.empty:
            errors.append(f"{group_column} appears in multiple splits: {sorted(leaking.index.astype(str))}")

    if errors:
        raise ValueError("; ".join(errors))
    return errors
