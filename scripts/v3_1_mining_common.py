from __future__ import annotations

import hashlib
import math
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def display_path(path: str | Path, root: Path = ROOT) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(value.resolve())


def stable_hash(*parts: object, length: int = 12) -> str:
    text = "|".join(str(part) for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    return f"{prefix}_{stable_hash(*parts, length=length)}"


def safe_token(value: object, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    text = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pcm_path(path: Path) -> bool:
    return path.suffix.lower() in {".pcm", ".raw"}


def audio_info(path: Path, pcm_defaults: dict[str, Any]) -> dict[str, int | float | str]:
    if is_pcm_path(path):
        sample_rate = int(pcm_defaults.get("sample_rate", 16000))
        channels = int(pcm_defaults.get("channels", 1))
        sample_width = int(pcm_defaults.get("sample_width_bytes", 2))
        frames = path.stat().st_size // max(channels * sample_width, 1)
        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": round(frames / float(sample_rate), 6) if sample_rate else 0.0,
            "format": str(pcm_defaults.get("encoding", "s16le")),
        }

    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_sec": round(float(info.frames) / float(info.samplerate), 6),
            "format": str(info.format),
        }
    except Exception:
        with wave.open(str(path), "rb") as f:
            sample_rate = int(f.getframerate())
            frames = int(f.getnframes())
            return {
                "sample_rate": sample_rate,
                "channels": int(f.getnchannels()),
                "duration_sec": round(float(frames) / float(sample_rate), 6),
                "format": "WAV",
            }


def load_audio_mono(path: Path, target_sample_rate: int, pcm_defaults: dict[str, Any] | None = None) -> np.ndarray:
    if is_pcm_path(path):
        defaults = pcm_defaults or {}
        sample_rate = int(defaults.get("sample_rate", target_sample_rate))
        channels = int(defaults.get("channels", 1))
        sample_width = int(defaults.get("sample_width_bytes", 2))
        encoding = str(defaults.get("encoding", "s16le")).lower()
        if sample_rate != target_sample_rate:
            raise ValueError(
                f"PCM sample_rate={sample_rate} differs from target_sample_rate={target_sample_rate}: {path}"
            )
        if sample_width != 2 or encoding != "s16le":
            raise ValueError(f"Unsupported PCM format for {path}: encoding={encoding} sample_width={sample_width}")
        data = np.fromfile(path, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            frames = data.size // channels
            data = data[: frames * channels].reshape(frames, channels).mean(axis=1)
        return data.astype(np.float32, copy=False)

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required to load WAV files for v3.1 mining") from exc
    audio, _ = librosa.load(path, sr=target_sample_rate, mono=True)
    return audio.astype(np.float32, copy=False)


def iter_window_ranges(audio_size: int, sample_rate: int, window_sec: float, hop_sec: float) -> list[tuple[int, int]]:
    window_size = int(round(sample_rate * window_sec))
    hop_size = int(round(sample_rate * hop_sec))
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_sec and hop_sec must be positive")
    if audio_size <= 0:
        return []
    if audio_size <= window_size:
        return [(0, audio_size)]
    return [(start, start + window_size) for start in range(0, audio_size - window_size + 1, hop_size)]


def pad_window(chunk: np.ndarray, window_size: int) -> np.ndarray:
    if chunk.size >= window_size:
        return chunk[:window_size].astype(np.float32, copy=False)
    return np.pad(chunk, (0, window_size - chunk.size)).astype(np.float32, copy=False)


def rms_energy(chunk: np.ndarray) -> tuple[float, float]:
    if chunk.size == 0:
        return 0.0, 0.0
    squared = np.square(chunk.astype(np.float32, copy=False))
    energy = float(np.sum(squared))
    rms = float(math.sqrt(float(np.mean(squared))))
    return energy, rms


def row_key(audio_file: object, start_time: object, end_time: object) -> str:
    return f"{audio_file}|{float(start_time):.6f}|{float(end_time):.6f}"
