from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import log_mel_feature


def test_board_compatible_log_mel_shape_and_finiteness() -> None:
    pytest.importorskip("librosa")
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    feature = log_mel_feature(
        audio,
        {
            "sample_rate": sample_rate,
            "n_fft": 1024,
            "hop_length": 160,
            "n_mels": 40,
            "fmin": 50,
            "fmax": 7600,
            "htk": True,
            "center": True,
            "top_db": 80,
            "normalize": "per_window_zscore",
        },
    )

    assert feature.shape == (40, 101)
    assert np.isfinite(feature).all()
