from __future__ import annotations

import random

import pandas as pd


def assign_group_splits(
    frame: pd.DataFrame,
    group_column: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if group_column not in frame.columns:
        raise ValueError(f"Missing group split column: {group_column}")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    groups = sorted(str(value) for value in frame[group_column].dropna().unique())
    if len(groups) < 3:
        raise ValueError(
            "Need at least 3 unique person_id values for train/val/test grouped split"
        )

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])
    if not test_groups:
        raise ValueError("Grouped split produced an empty test set")

    def choose_split(group: object) -> str:
        group_id = str(group)
        if group_id in train_groups:
            return "train"
        if group_id in val_groups:
            return "val"
        return "test"

    output = frame.copy()
    output["split"] = output[group_column].map(choose_split)
    return output
