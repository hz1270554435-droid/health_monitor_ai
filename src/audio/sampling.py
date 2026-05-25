from __future__ import annotations

import numpy as np
import pandas as pd


WEIGHTED_SAMPLER_MODES = {"balanced", "hard_negative_oversample", "source_weighted", "weighted_random"}


def config_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"1", "true", "yes", "y", "on"}


def class_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {int(key): int(value) for key, value in frame["label_id"].value_counts().sort_index().items()}


def inverse_class_weights(frame: pd.DataFrame, num_classes: int = 2) -> list[float]:
    counts = class_counts(frame)
    total = sum(counts.values())
    weights: list[float] = []
    for label_id in range(num_classes):
        count = counts.get(label_id, 0)
        weights.append(float(total / (num_classes * count)) if count else 0.0)
    return weights


def row_sample_weights(frame: pd.DataFrame, weight_column: str = "sample_weight") -> np.ndarray:
    if weight_column not in frame.columns:
        raise ValueError(f"Sample weight column not found in manifest: {weight_column}")
    values = pd.to_numeric(frame[weight_column], errors="coerce")
    missing = values.isna()
    if missing.any():
        bad_rows = frame.loc[missing, "clip_id"].astype(str).head(10).tolist() if "clip_id" in frame.columns else []
        raise ValueError(f"Sample weight column contains non-numeric values: {weight_column}; examples={bad_rows}")
    non_positive = values <= 0
    if non_positive.any():
        bad_rows = (
            frame.loc[non_positive, "clip_id"].astype(str).head(10).tolist()
            if "clip_id" in frame.columns
            else []
        )
        raise ValueError(f"Sample weight column contains values <= 0: {weight_column}; examples={bad_rows}")
    return values.to_numpy(dtype=np.float64)


def sample_weights(frame: pd.DataFrame, sampling: dict | None = None) -> np.ndarray:
    cfg = sampling or {}
    mode = str(cfg.get("mode", "none")).strip().lower()
    weight_column = str(cfg.get("weight_column", "sample_weight")).strip() or "sample_weight"
    use_sample_weight = config_bool(cfg.get("use_sample_weight"), default=False)
    weights = np.ones(len(frame), dtype=np.float64)

    if mode == "weighted_random":
        if not use_sample_weight:
            raise ValueError("sampling.mode=weighted_random requires sampling.use_sample_weight=true")
        return row_sample_weights(frame, weight_column=weight_column)

    if mode in {"balanced", "hard_negative_oversample", "source_weighted"}:
        class_weight = inverse_class_weights(frame, num_classes=int(frame["label_id"].max()) + 1)
        weights *= frame["label_id"].map(lambda value: class_weight[int(value)]).to_numpy(dtype=np.float64)

    if mode == "hard_negative_oversample" or bool(cfg.get("hard_negative_oversample", False)):
        multiplier = float(cfg.get("hard_negative_multiplier", 2.0))
        hard = frame.get("hard_negative", False)
        hard_values = pd.Series(hard).map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        weights *= np.where(hard_values.to_numpy(dtype=bool), multiplier, 1.0)

    if mode == "source_weighted":
        source_counts = frame.get("source_dataset", pd.Series(["unknown"] * len(frame))).value_counts()
        source_weight = source_counts.map(lambda count: 1.0 / max(float(count), 1.0)).to_dict()
        weights *= frame.get("source_dataset", pd.Series(["unknown"] * len(frame))).map(source_weight).to_numpy()

    if use_sample_weight:
        weights *= row_sample_weights(frame, weight_column=weight_column)
    return weights


def build_weighted_sampler(frame: pd.DataFrame, sampling: dict | None, torch_module):
    cfg = sampling or {}
    mode = str(cfg.get("mode", "none")).strip().lower()
    if mode not in WEIGHTED_SAMPLER_MODES:
        return None
    weights = sample_weights(frame, cfg)
    return torch_module.utils.data.WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )


def loss_class_weights(frame: pd.DataFrame, training: dict, torch_module, device):
    value = training.get("class_weight")
    if value in (None, "", "none", False):
        return None
    if str(value).strip().lower() not in {"auto", "balanced", "class_weight"}:
        return None
    weights = inverse_class_weights(frame, num_classes=int(frame["label_id"].max()) + 1)
    return torch_module.tensor(weights, dtype=torch_module.float32, device=device)
