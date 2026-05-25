from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.train import run_training


def _write_feature(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, feature=np.full((40, 101), value, dtype=np.float32))


def test_audio_task_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rows = []
    splits = ["train", "train", "val", "val", "test", "test"]
    labels = [0, 1, 0, 1, 0, 1]
    for index, (split, label_id) in enumerate(zip(splits, labels)):
        feature_path = tmp_path / "processed" / "features" / f"{index}.npz"
        _write_feature(feature_path, float(label_id))
        rows.append(
            {
                "clip_id": f"clip_{index}",
                "person_id": f"P{index}",
                "split_group": f"P{index}",
                "split": split,
                "label_id": label_id,
                "label": "snore" if label_id else "non_snore",
                "feature_path": f"features/{index}.npz",
                "window_index": 0,
                "window_start": 0.0,
                "window_end": 1.0,
                "audio_file": f"{index}.wav",
            }
        )
    manifest_path = tmp_path / "processed" / "audio_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    config = {
        "project": {"seed": 42},
        "paths": {"processed_dir": "processed", "models_dir": "models", "results_dir": "results"},
        "labels": {"negative": "non_snore", "positive": "snore", "class_order": ["non_snore", "snore"]},
        "split": {"group_column": "split_group", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "training": {
            "model": "small_cnn",
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "num_workers": 0,
            "device": "cpu",
            "early_stopping_patience": 0,
            "optimizer": "adamw",
            "class_weight": "auto",
        },
        "eval": {"primary_metric": "val_f1"},
        "run": {"name": "unit_snore"},
    }

    paths = run_training(config, manifest_path=manifest_path, run_name="unit_run", epochs=1, root=tmp_path)

    assert paths["checkpoint_path"].exists()
    assert (tmp_path / "results" / "unit_run" / "metrics.json").exists()
