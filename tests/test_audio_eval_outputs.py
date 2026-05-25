from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.evaluate import evaluate_checkpoint
from src.audio.models import build_model


def test_audio_task_eval_writes_generic_probability_outputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    processed = tmp_path / "processed"
    feature_dir = processed / "features"
    feature_dir.mkdir(parents=True)
    rows = []
    for index, (split, label_id) in enumerate(
        [("train", 0), ("train", 1), ("val", 0), ("val", 1), ("test", 0), ("test", 1)]
    ):
        np.savez_compressed(feature_dir / f"{index}.npz", feature=np.full((40, 101), label_id, dtype=np.float32))
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
    manifest_path = processed / "audio_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    model = build_model("small_cnn", num_classes=2)
    model_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": ["non_snore", "snore"],
            "model_name": "small_cnn",
        },
        model_path,
    )
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
        },
        "run": {"name": "unit_snore"},
        "eval": {"false_alarm": {"event_merge_gap_seconds": 5.0, "night_hours": 8.0}},
    }

    metrics = evaluate_checkpoint(
        config,
        run_dir=tmp_path / "eval",
        model_path=model_path,
        manifest_path=manifest_path,
        split="test",
        threshold=0.5,
        root=tmp_path,
    )

    assert metrics["positive_probability_column"] == "snore_prob"
    assert (tmp_path / "eval" / "metrics_threshold_0p50.json").exists()
    assert (tmp_path / "eval" / "misclassified_threshold_0p50.csv").exists()
