from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.manifest import canonicalize_manifest
from src.audio.task_views import build_non_cough_pool, build_task_view


def test_build_snore_task_view_reuses_non_cough_pool_and_maps_cough_negative() -> None:
    manifest = canonicalize_manifest(
        pd.DataFrame(
            [
                {
                    "clip_id": "snore_1",
                    "session_id": "S1",
                    "person_id": "P1",
                    "source_dataset": "snore_set",
                    "audio_path": "snore.wav",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "original_label": "snore",
                    "normalized_label": "snore",
                    "quality": "reviewed",
                    "split_group": "P1",
                    "split": "train",
                    "source_audio_id": "A1",
                },
                {
                    "clip_id": "cough_1",
                    "session_id": "S2",
                    "person_id": "P2",
                    "source_dataset": "cough_set",
                    "audio_path": "cough.wav",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "original_label": "cough",
                    "normalized_label": "cough",
                    "quality": "reviewed",
                    "split_group": "P2",
                    "split": "val",
                    "source_audio_id": "A2",
                },
                {
                    "clip_id": "noise_1",
                    "session_id": "S3",
                    "person_id": "P3",
                    "source_dataset": "negative_set",
                    "audio_path": "noise.wav",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "original_label": "non_cough",
                    "normalized_label": "non_cough",
                    "quality": "reviewed",
                    "split_group": "P3",
                    "split": "test",
                    "source_audio_id": "A3",
                },
            ]
        )
    )
    pool = build_non_cough_pool(manifest, {"non_cough_pool": {"normalized_labels": ["non_cough"]}})
    config = {
        "project": {"seed": 42},
        "task": {
            "name": "snore_v1",
            "positive_label": "snore",
            "negative_label": "non_snore",
            "positive_source_labels": ["snore"],
            "negative_source_labels": ["non_cough", "cough"],
            "negative_pool": {"include": True},
        },
        "labels": {"positive": "snore", "negative": "non_snore"},
        "split": {"group_column": "split_group", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
    }

    view = build_task_view(manifest, config, non_cough_pool=pool)

    assert set(view["task_label"]) == {"snore", "non_snore"}
    assert view.set_index("clip_id").loc["snore_1", "label_id"] == 1
    assert view.set_index("clip_id").loc["cough_1", "label_id"] == 0
    assert view.set_index("clip_id").loc["noise_1", "negative_role"] == "shared_negative"
