from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.task_views import assign_task_splits


def test_assign_task_splits_keeps_unknown_external_train_only() -> None:
    frame = pd.DataFrame(
        {
            "clip_id": ["a", "b", "c"],
            "split_group": ["P1", "P1", "UNKNOWN_A"],
            "split": ["train", "", ""],
            "eval_allowed": [True, True, False],
        }
    )
    config = {
        "project": {"seed": 42},
        "split": {"group_column": "split_group", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }

    output = assign_task_splits(frame, config)

    assert output.set_index("clip_id").loc["b", "split"] == "train"
    assert output.set_index("clip_id").loc["c", "split"] == "train"
    assert output.groupby("split_group")["split"].nunique().max() == 1
