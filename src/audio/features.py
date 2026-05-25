from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio_segment(
    path: str | Path,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for audio preprocessing") from exc

    duration = max(0.0, float(end_time) - float(start_time))
    if duration <= 0:
        raise ValueError(f"Invalid audio segment: start={start_time}, end={end_time}")
    audio, _ = librosa.load(
        Path(path),
        sr=sample_rate,
        mono=True,
        offset=float(start_time),
        duration=duration,
    )
    return audio.astype(np.float32, copy=False)


def iter_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[int, int, np.ndarray]]:
    window_size = int(round(sample_rate * window_seconds))
    hop_size = int(round(sample_rate * hop_seconds))
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("Window and hop sizes must be positive")
    if audio.size == 0:
        return []

    windows: list[tuple[int, int, np.ndarray]] = []
    max_start = max(0, audio.size - window_size)
    starts = list(range(0, max_start + 1, hop_size))
    if not starts:
        starts = [0]
    for start in starts:
        end = start + window_size
        chunk = audio[start:end]
        if chunk.size < window_size:
            chunk = np.pad(chunk, (0, window_size - chunk.size))
        windows.append((start, end, chunk.astype(np.float32, copy=False)))
    return windows


def log_mel_feature(audio: np.ndarray, config: dict) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for Log-Mel feature extraction") from exc

    sample_rate = int(config["sample_rate"])
    mel_scale = str(config.get("mel_scale", "")).strip().lower()
    htk = bool(config.get("htk", mel_scale == "htk"))
    center = bool(config.get("center", True))
    top_db = config.get("top_db", 80)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=int(config.get("n_fft", 1024)),
        hop_length=int(config.get("hop_length", 160)),
        n_mels=int(config.get("n_mels", 40)),
        fmin=float(config.get("fmin", 50)),
        fmax=float(config.get("fmax", sample_rate / 2)),
        power=2.0,
        htk=htk,
        center=center,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=top_db).astype(np.float32)
    normalize = str(config.get("normalize", "per_window_zscore")).strip().lower()
    if normalize in {"none", "false", "no"}:
        return log_mel
    if normalize != "per_window_zscore":
        raise ValueError(f"Unsupported audio normalization mode: {normalize}")
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    return (log_mel - mean) / (std + 1e-6)
