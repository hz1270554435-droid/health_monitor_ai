from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.dataset import FeatureDataset


def test_feature_dataset_loads_one_feature(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    np.savez_compressed(feature_dir / "one.npz", feature=np.zeros((40, 101), dtype=np.float32))
    manifest = pd.DataFrame({"feature_path": ["features/one.npz"], "label_id": [1]})

    dataset = FeatureDataset(manifest, root=tmp_path).unwrap()
    feature, label = dataset[0]

    assert tuple(feature.shape) == (1, 40, 101)
    assert int(label) == 1
