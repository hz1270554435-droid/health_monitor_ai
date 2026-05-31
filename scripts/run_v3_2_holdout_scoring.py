from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.audio.features import log_mel_feature
from src.audio.models import build_model
from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import load_audio_mono, pad_window, resolve_path

THRESHOLDS = [0.30, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
AUDIO_SOURCE_FIELDS = ["original_path", "source_audio", "audio_file"]
NOT_EXECUTED = [
    "training",
    "ONNX export",
    "deployment export",
    "board package generation",
    "training label modification",
    "old v3.1/v3 model modification",
    "raw WAV/JSON modification",
    "ml/data/raw write",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v3.2 holdout scoring on the canonical window manifest.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_2_recall_fix.yaml")
    parser.add_argument(
        "--checkpoint",
        default="models/audio_v3_2_recall_fix/audio_baseline_v3_2_recall_fix_train1/best_model.pt",
    )
    parser.add_argument("--manifest", default="review/v3_2_recall_fix/window_manifest_combined.csv")
    parser.add_argument(
        "--v3-1-scored",
        default="review/v3_2_recall_fix/v3_1_fpfix_onnx_scored_manifest_combined_corrected.csv",
    )
    parser.add_argument(
        "--report",
        default="review/v3_2_recall_fix/v3_2_train1_holdout_scoring_report.json",
    )
    parser.add_argument(
        "--summary-md",
        default="review/v3_2_recall_fix/v3_2_train1_holdout_scoring_summary.md",
    )
    parser.add_argument(
        "--threshold-sweep-csv",
        default="review/v3_2_recall_fix/v3_2_train1_holdout_threshold_sweep.csv",
    )
    parser.add_argument(
        "--comparison-csv",
        default="review/v3_2_recall_fix/v3_2_train1_vs_v3_1_holdout_comparison.csv",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def path_arg(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return frame


def normalize_path_key(value: object) -> str:
    return os.path.normcase(os.path.normpath(str(value).strip()))


def normalize_time_key(value: object) -> str:
    return f"{round(float(value), 3):.3f}"


def normalized_join_key(row: pd.Series) -> str:
    return "|".join(
        [
            normalize_path_key(row["original_path"]),
            normalize_time_key(row["window_start_sec"]),
            normalize_time_key(row["window_end_sec"]),
        ]
    )


def expected_input_shape(audio_config: dict[str, Any]) -> list[int]:
    sample_rate = int(audio_config["sample_rate"])
    window_samples = int(round(sample_rate * float(audio_config["window_seconds"])))
    hop_length = int(audio_config.get("hop_length", 160))
    n_fft = int(audio_config.get("n_fft", 1024))
    n_mels = int(audio_config.get("n_mels", 40))
    center = bool(audio_config.get("center", True))
    if center:
        time_bins = 1 + window_samples // hop_length
    else:
        time_bins = 1 + max(0, (window_samples - n_fft) // hop_length)
    return [1, 1, n_mels, time_bins]


def frontend_summary(config: dict[str, Any]) -> dict[str, Any]:
    audio = config["audio"]
    class_order = list(config["labels"].get("class_order") or [config["labels"]["negative"], config["labels"]["positive"]])
    checks = {
        "frontend": audio.get("frontend") == "board_htk_no_norm_v1",
        "sample_rate": int(audio.get("sample_rate")) == 16000,
        "mono": bool(audio.get("mono")) is True,
        "window_seconds": float(audio.get("window_seconds")) == 1.0,
        "hop_seconds": float(audio.get("hop_seconds")) == 0.5,
        "n_mels": int(audio.get("n_mels")) == 40,
        "mel_scale_htk": str(audio.get("mel_scale")).strip().lower() == "htk" and bool(audio.get("htk")) is True,
        "normalize_none": str(audio.get("normalize")).strip().lower() == "none",
        "class_order": class_order == ["non_cough", "cough"],
        "expected_input_shape": expected_input_shape(audio) == [1, 1, 40, 101],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "summary": {
            "sample_rate": int(audio.get("sample_rate")),
            "channels": "mono",
            "window_sec": float(audio.get("window_seconds")),
            "hop_sec": float(audio.get("hop_seconds")),
            "feature": f"{int(audio.get('n_mels'))}-bin HTK Log-Mel",
            "normalization": audio.get("normalize"),
            "class_order": class_order,
            "expected_input_shape": expected_input_shape(audio),
        },
    }


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for v3.2 holdout scoring") from exc
    return torch


def load_checkpoint(torch, checkpoint_path: Path, config: dict[str, Any], device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    training_cfg = checkpoint_config.get("training", {}) if isinstance(checkpoint_config, dict) else {}
    config_training = config.get("training", {})
    model_name = str(training_cfg.get("model") or config_training.get("model", "small_cnn")).strip().lower()
    class_names = list(
        checkpoint.get("class_names")
        or config["labels"].get("class_order")
        or [config["labels"]["negative"], config["labels"]["positive"]]
    )
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, model_name


def choose_audio_path(row: pd.Series, counts: dict[str, int]) -> tuple[Path, str]:
    for field in AUDIO_SOURCE_FIELDS:
        value = row.get(field, "")
        if pd.notna(value) and str(value).strip():
            candidate = resolve_path(str(value).strip())
            if candidate.exists():
                counts[field] = counts.get(field, 0) + 1
                return candidate, field
    raise FileNotFoundError(
        "No readable audio source found via original_path/source_audio/audio_file; "
        f"row key={row.get('_join_key')}"
    )


def quantiles(series: pd.Series, include_min_max: bool = True) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        keys = ["p50", "p90", "p95"] + (["min", "max"] if include_min_max else [])
        return {key: None for key in keys}
    result: dict[str, float | None] = {
        "p50": float(numeric.quantile(0.50)),
        "p90": float(numeric.quantile(0.90)),
        "p95": float(numeric.quantile(0.95)),
    }
    if include_min_max:
        result = {"min": float(numeric.min()), **result, "max": float(numeric.max())}
    return result


def score_v3_2(rows: pd.DataFrame, config: dict[str, Any], checkpoint_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, model_name = load_checkpoint(torch, checkpoint_path, config, device)
    if class_names != ["non_cough", "cough"]:
        raise ValueError(f"Unexpected class_names: {class_names}")
    expected_shape = expected_input_shape(config["audio"])
    cough_index = int(class_names.index("cough"))
    non_cough_index = int(class_names.index("non_cough"))
    sample_rate = int(config["audio"]["sample_rate"])
    window_size = int(round(sample_rate * float(config["audio"]["window_seconds"])))
    audio_cache: dict[str, np.ndarray] = {}
    audio_source_counts = {field: 0 for field in AUDIO_SOURCE_FIELDS}
    observed_shapes: list[list[int]] = []
    cough_probs: list[float] = []
    non_cough_probs: list[float] = []
    audio_paths: list[str] = []
    audio_source_fields: list[str] = []

    for _, row in rows.iterrows():
        audio_path, source_field = choose_audio_path(row, audio_source_counts)
        cache_key = str(audio_path)
        audio = audio_cache.get(cache_key)
        if audio is None:
            audio = load_audio_mono(audio_path, sample_rate, {})
            audio_cache[cache_key] = audio
        start_time = float(row["window_start_sec"])
        end_time = float(row["window_end_sec"])
        start_sample = max(0, int(round(start_time * sample_rate)))
        end_sample = int(round(end_time * sample_rate))
        if end_sample <= start_sample:
            end_sample = start_sample + window_size
        chunk = pad_window(audio[start_sample:end_sample], window_size)
        feature = log_mel_feature(chunk, config["audio"])
        tensor = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
        observed_shape = [int(value) for value in tensor.shape]
        if observed_shape != expected_shape:
            raise ValueError(f"Observed input shape {observed_shape}, expected {expected_shape}")
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        non_cough_prob = float(probs[non_cough_index])
        cough_prob = float(probs[cough_index])
        non_cough_probs.append(non_cough_prob)
        cough_probs.append(cough_prob)
        observed_shapes.append(observed_shape)
        audio_paths.append(str(audio_path))
        audio_source_fields.append(source_field)

    output = rows.copy()
    output["v3_2_audio_source_field"] = audio_source_fields
    output["v3_2_audio_path"] = audio_paths
    output["v3_2_non_cough_prob"] = non_cough_probs
    output["v3_2_cough_prob"] = cough_probs
    output["delta_cough_prob"] = pd.to_numeric(output["v3_2_cough_prob"]) - pd.to_numeric(output["fpfix_cough_prob"])
    output["delta_non_cough_prob"] = pd.to_numeric(output["v3_2_non_cough_prob"]) - pd.to_numeric(output["fpfix_non_cough_prob"])
    return output, {
        "resolved_model_name": model_name,
        "class_order": class_names,
        "expected_input_shape": expected_shape,
        "observed_input_shapes": sorted({tuple(shape) for shape in observed_shapes}),
        "device_used": str(device),
        "audio_path_resolution_counts": audio_source_counts,
        "audio_files_loaded": int(len(audio_cache)),
        "audio_source_policy": "original_path -> source_audio -> audio_file",
    }


def validate_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    required = [
        "original_path",
        "audio_file",
        "person_id",
        "scenario_canonical",
        "split_role_suggested",
        "fs",
        "channels",
        "window_start_sec",
        "window_end_sec",
    ]
    missing = [name for name in required if name not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    raw_root = (ROOT / "data" / "raw").resolve()
    raw_hits: list[str] = []
    for value in manifest["audio_file"].astype(str).unique().tolist():
        resolved = resolve_path(value).resolve()
        if resolved == raw_root or raw_root in resolved.parents:
            raw_hits.append(value)
    if raw_hits:
        raise ValueError(f"Manifest references ml/data/raw paths: {raw_hits[:10]}")
    return {
        "rows": int(len(manifest)),
        "unique_audio_files": int(manifest["audio_file"].nunique()),
        "fs_counts": {str(k): int(v) for k, v in manifest["fs"].value_counts().sort_index().items()},
        "channel_counts": {str(k): int(v) for k, v in manifest["channels"].value_counts().sort_index().items()},
        "ml_data_raw_hits": raw_hits,
    }


def attach_v3_1_scores(manifest: pd.DataFrame, v3_1_scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "original_path",
        "window_start_sec",
        "window_end_sec",
        "fpfix_non_cough_prob",
        "fpfix_cough_prob",
        "person_id_for_analysis",
        "source_actor_type",
        "person_analysis_included",
    ]
    missing = [name for name in required if name not in v3_1_scored.columns]
    if missing:
        raise ValueError(f"v3.1 scored manifest missing columns: {missing}")

    base = manifest.copy()
    base["_join_key"] = base.apply(normalized_join_key, axis=1)
    ref = v3_1_scored.copy()
    ref["_join_key"] = ref.apply(normalized_join_key, axis=1)
    duplicate_ref = int(ref.duplicated("_join_key").sum())
    if duplicate_ref:
        raise ValueError(f"v3.1 scored manifest contains {duplicate_ref} duplicate normalized keys")

    keep_columns = list(dict.fromkeys(list(manifest.columns) + [
        "_join_key",
        "fpfix_non_cough_prob",
        "fpfix_cough_prob",
        "score_model_name",
        "score_frontend",
        "score_source",
        "person_id_raw",
        "person_id_for_analysis",
        "source_actor_type",
        "operator_id",
        "environment_id",
        "person_analysis_included",
    ]))
    joined = base.merge(ref[keep_columns], on="_join_key", how="left", suffixes=("", "_v3_1"))
    missing_score_rows = int(joined["fpfix_cough_prob"].isna().sum())
    join_report = {
        "window_manifest_rows": int(len(base)),
        "v3_1_scored_rows": int(len(ref)),
        "joined_rows": int(len(joined)),
        "missing_score_rows": missing_score_rows,
        "duplicate_v3_1_keys": duplicate_ref,
        "strict_validation": {
            "scored_rows_eq_window_manifest_rows": int(len(base)) == int(len(manifest)),
            "joined_rows_eq_scored_rows": int(len(joined)) == int(len(base)),
            "missing_score_rows_eq_0": missing_score_rows == 0,
        },
    }
    if missing_score_rows != 0:
        raise ValueError(f"Strict join validation failed: missing_score_rows={missing_score_rows}")
    return joined, join_report


def build_truth_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    scenario = output["scenario_canonical"].astype(str)
    output["proxy_positive"] = scenario.eq("cough")
    output["proxy_negative"] = scenario.isin(["mixed_non_cough", "quiet", "speech", "throat_clear", "sniff"])
    output["proxy_truth_available"] = output["proxy_positive"] | output["proxy_negative"]
    output["proxy_truth_label"] = np.where(output["proxy_positive"], "cough", np.where(output["proxy_negative"], "non_cough", ""))
    output["metric_label"] = "proxy_metric"
    output["manual_truth_available"] = False
    output["manual_truth_label"] = ""
    return output


def confusion_counts(frame: pd.DataFrame, score_column: str, threshold: float, truth_column: str) -> dict[str, Any]:
    if frame.empty:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "rows": 0}
    y_true = frame[truth_column].astype(str).eq("cough").astype(int)
    y_pred = pd.to_numeric(frame[score_column], errors="coerce").ge(threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rows": int(len(frame)),
    }


def event_count(flags: list[bool]) -> int:
    count = 0
    previous = False
    for value in flags:
        if value and not previous:
            count += 1
        previous = value
    return count


def event_level_fp_hour(frame: pd.DataFrame, score_column: str, threshold: float) -> dict[str, Any]:
    quiet = frame[frame["split_role_suggested"].astype(str).eq("long_quiet_eval")].copy()
    if quiet.empty:
        return {
            "threshold": threshold,
            "window_level_fp_windows": 0,
            "event_level_fp_events": 0,
            "duration_hours": 0.0,
            "window_level_fp_per_hour": None,
            "event_level_fp_per_hour": None,
        }
    quiet["trigger"] = pd.to_numeric(quiet[score_column], errors="coerce").ge(threshold)
    duration_hours = float(pd.to_numeric(quiet["duration_s"], errors="coerce").dropna().sum() / 3600.0)
    grouped_events = 0
    for _, group in quiet.groupby(["audio_file", "session_id"], dropna=False):
        group = group.sort_values(["window_start_sec", "window_end_sec"])
        grouped_events += event_count(group["trigger"].astype(bool).tolist())
    fp_windows = int(quiet["trigger"].sum())
    return {
        "threshold": threshold,
        "window_level_fp_windows": fp_windows,
        "event_level_fp_events": int(grouped_events),
        "duration_hours": duration_hours,
        "window_level_fp_per_hour": (fp_windows / duration_hours) if duration_hours > 0 else None,
        "event_level_fp_per_hour": (grouped_events / duration_hours) if duration_hours > 0 else None,
    }


def activation_rate(frame: pd.DataFrame, score_column: str, threshold: float) -> dict[str, Any]:
    rows = int(len(frame))
    activations = int(pd.to_numeric(frame[score_column], errors="coerce").ge(threshold).sum()) if rows else 0
    return {
        "rows": rows,
        "activations": activations,
        "activation_rate": (activations / rows) if rows else None,
    }


def comparison_summary(frame: pd.DataFrame, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if group_column not in frame.columns:
        return rows
    for value, subset in frame.groupby(group_column, dropna=False):
        rows.append(
            {
                group_column: "" if pd.isna(value) else str(value),
                "rows": int(len(subset)),
                "fpfix_cough_prob": quantiles(subset["fpfix_cough_prob"]),
                "v3_2_cough_prob": quantiles(subset["v3_2_cough_prob"]),
                "delta_cough_prob": quantiles(subset["delta_cough_prob"], include_min_max=False),
                "delta_mean": float(pd.to_numeric(subset["delta_cough_prob"], errors="coerce").mean()),
                "delta_positive_rows": int(pd.to_numeric(subset["delta_cough_prob"], errors="coerce").gt(0).sum()),
                "delta_negative_rows": int(pd.to_numeric(subset["delta_cough_prob"], errors="coerce").lt(0).sum()),
            }
        )
    return rows


def build_threshold_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_specs = [
        ("scenario_canonical", "scenario_canonical"),
        ("split_role_suggested", "split_role_suggested"),
        ("person_id_for_analysis", "person_id_for_analysis"),
    ]

    for threshold in THRESHOLDS:
        for group_kind, group_column in group_specs:
            for group_value, subset in frame.groupby(group_column, dropna=False):
                name = "" if pd.isna(group_value) else str(group_value)
                proxy_subset = subset[subset["proxy_truth_available"]].copy()
                v3_1 = confusion_counts(proxy_subset, "fpfix_cough_prob", threshold, "proxy_truth_label")
                v3_2 = confusion_counts(proxy_subset, "v3_2_cough_prob", threshold, "proxy_truth_label")
                rows.append(
                    {
                        "metric_kind": "proxy_metric",
                        "threshold": threshold,
                        "group_kind": group_kind,
                        "group_value": name,
                        "rows": int(len(proxy_subset)),
                        "v3_1_tp": v3_1["tp"],
                        "v3_1_fp": v3_1["fp"],
                        "v3_1_tn": v3_1["tn"],
                        "v3_1_fn": v3_1["fn"],
                        "v3_1_precision": v3_1["precision"],
                        "v3_1_recall_or_proxy": v3_1["recall"],
                        "v3_1_f1": v3_1["f1"],
                        "v3_2_tp": v3_2["tp"],
                        "v3_2_fp": v3_2["fp"],
                        "v3_2_tn": v3_2["tn"],
                        "v3_2_fn": v3_2["fn"],
                        "v3_2_precision": v3_2["precision"],
                        "v3_2_recall_or_proxy": v3_2["recall"],
                        "v3_2_f1": v3_2["f1"],
                        "delta_fp": v3_2["fp"] - v3_1["fp"],
                        "delta_fn": v3_2["fn"] - v3_1["fn"],
                        "delta_recall_or_proxy": v3_2["recall"] - v3_1["recall"],
                        "delta_precision": v3_2["precision"] - v3_1["precision"],
                        "notes": "scenario-hint proxy metrics; not deployment threshold evidence",
                    }
                )
    return pd.DataFrame(rows)


def rank_fp_sources(frame: pd.DataFrame, threshold: float) -> list[dict[str, Any]]:
    proxy_negative = frame[frame["proxy_negative"]].copy()
    rows: list[dict[str, Any]] = []
    for scenario, subset in proxy_negative.groupby("scenario_canonical", dropna=False):
        v3_1_fp = int(pd.to_numeric(subset["fpfix_cough_prob"], errors="coerce").ge(threshold).sum())
        v3_2_fp = int(pd.to_numeric(subset["v3_2_cough_prob"], errors="coerce").ge(threshold).sum())
        rows.append(
            {
                "scenario_canonical": "" if pd.isna(scenario) else str(scenario),
                "rows": int(len(subset)),
                "v3_1_fp": v3_1_fp,
                "v3_2_fp": v3_2_fp,
                "delta_fp": v3_2_fp - v3_1_fp,
                "v3_2_max": float(pd.to_numeric(subset["v3_2_cough_prob"], errors="coerce").max()),
                "v3_2_p95": float(pd.to_numeric(subset["v3_2_cough_prob"], errors="coerce").quantile(0.95)),
            }
        )
    rows.sort(key=lambda item: (-item["v3_2_fp"], -item["delta_fp"], item["scenario_canonical"]))
    return rows


def compute_proxy_findings(frame: pd.DataFrame) -> dict[str, Any]:
    p01 = frame[
        frame["person_id_for_analysis"].astype(str).eq("p01")
        & frame["scenario_canonical"].astype(str).eq("cough")
        & frame["person_analysis_included"].astype(bool)
    ]
    p02 = frame[
        frame["person_id_for_analysis"].astype(str).eq("p02")
        & frame["scenario_canonical"].astype(str).eq("cough")
        & frame["person_analysis_included"].astype(bool)
    ]
    p03 = frame[
        frame["person_id_for_analysis"].astype(str).eq("p03")
        & frame["scenario_canonical"].astype(str).eq("cough")
        & frame["person_analysis_included"].astype(bool)
    ]
    findings = {
        "p01_cough_session_activation_rate": {
            "threshold_0_50": {
                "v3_1": activation_rate(p01, "fpfix_cough_prob", 0.50),
                "v3_2": activation_rate(p01, "v3_2_cough_prob", 0.50),
            },
            "threshold_0_60": {
                "v3_1": activation_rate(p01, "fpfix_cough_prob", 0.60),
                "v3_2": activation_rate(p01, "v3_2_cough_prob", 0.60),
            },
        },
        "p02_cough_session_activation_rate": {
            "threshold_0_50": {
                "v3_1": activation_rate(p02, "fpfix_cough_prob", 0.50),
                "v3_2": activation_rate(p02, "v3_2_cough_prob", 0.50),
            }
        },
        "p03_cough_session_activation_rate": {
            "threshold_0_50": {
                "v3_1": activation_rate(p03, "fpfix_cough_prob", 0.50),
                "v3_2": activation_rate(p03, "v3_2_cough_prob", 0.50),
            },
            "threshold_0_60": {
                "v3_1": activation_rate(p03, "fpfix_cough_prob", 0.60),
                "v3_2": activation_rate(p03, "v3_2_cough_prob", 0.60),
            },
        },
        "long_quiet_fp_hour": {
            "v3_1": [event_level_fp_hour(frame, "fpfix_cough_prob", threshold) for threshold in THRESHOLDS],
            "v3_2": [event_level_fp_hour(frame, "v3_2_cough_prob", threshold) for threshold in THRESHOLDS],
        },
    }
    return findings


def derive_recommendation(frame: pd.DataFrame, fp_ranking: list[dict[str, Any]], findings: dict[str, Any]) -> dict[str, Any]:
    p03_v32 = findings["p03_cough_session_activation_rate"]["threshold_0_50"]["v3_2"]["activation_rate"]
    p03_v31 = findings["p03_cough_session_activation_rate"]["threshold_0_50"]["v3_1"]["activation_rate"]
    long_quiet_v32 = next((row for row in findings["long_quiet_fp_hour"]["v3_2"] if math.isclose(row["threshold"], 0.50)), None)
    long_quiet_v31 = next((row for row in findings["long_quiet_fp_hour"]["v3_1"] if math.isclose(row["threshold"], 0.50)), None)
    mixed_row = next((row for row in fp_ranking if row["scenario_canonical"] == "mixed_non_cough"), None)
    speech_row = next((row for row in fp_ranking if row["scenario_canonical"] == "speech"), None)
    throat_row = next((row for row in fp_ranking if row["scenario_canonical"] == "throat_clear"), None)
    sniff_row = next((row for row in fp_ranking if row["scenario_canonical"] == "sniff"), None)

    recommend = "continue eval"
    reasons: list[str] = []
    sampler_adjustments: list[str] = []

    if long_quiet_v32 and long_quiet_v31 and long_quiet_v32["event_level_fp_per_hour"] is not None and long_quiet_v31["event_level_fp_per_hour"] is not None:
        if long_quiet_v32["event_level_fp_per_hour"] > long_quiet_v31["event_level_fp_per_hour"]:
            reasons.append("long_quiet event-level FP/hour worsened at threshold 0.50")
            recommend = "train2"
    if mixed_row and mixed_row["delta_fp"] > 0:
        reasons.append("mixed_non_cough increased false-positive windows")
        recommend = "train2"
        sampler_adjustments.append("increase weight for mixed_non_cough / handling_noise / knock hard negatives")
    for row, label in ((speech_row, "speech"), (throat_row, "throat_clear"), (sniff_row, "sniff")):
        if row and row["delta_fp"] > 0:
            reasons.append(f"{label} high-score non_cough windows increased")
            recommend = "train2"
            sampler_adjustments.append(f"increase weight for {label} negative windows or equivalent candidate role")
    if p03_v32 is not None and p03_v31 is not None and p03_v32 < p03_v31:
        reasons.append("p03 unseen-person cough-session activation rate dropped")
        recommend = "continue eval" if recommend != "train2" else recommend
    if recommend == "train2":
        sampler_adjustments.append("reduce over-emphasis of positive_recall_fix_candidate weights if proxy activation improved but FP sources worsened")

    if not reasons:
        recommend = "continue eval"
        reasons.append("proxy and FP evidence do not yet justify export; continue holdout evaluation only")

    return {
        "recommended_next_step": recommend,
        "reasons": reasons,
        "sampler_adjustment_direction": sampler_adjustments,
        "allowed_recommendations_only": True,
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    recommendation = report["recommendation"]
    findings = report["proxy_findings"]
    long_quiet_v32 = next((row for row in findings["long_quiet_fp_hour"]["v3_2"] if math.isclose(row["threshold"], 0.50)), None)
    long_quiet_v31 = next((row for row in findings["long_quiet_fp_hour"]["v3_1"] if math.isclose(row["threshold"], 0.50)), None)
    lines = [
        "# v3.2 Train1 Holdout Scoring Summary",
        "",
        f"- status: `{report['status']}`",
        f"- model: `{report['resolved_model_name']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- scored_rows: `{report['rows']['scored']}`",
        f"- joined_v3_1_rows: `{report['join_validation']['joined_rows']}`",
        f"- known_error_regression: `{report['known_error_regression']['status']}`",
        f"- recommended_next_step: `{recommendation['recommended_next_step']}`",
        "",
        "## Proxy Metrics",
        "",
        "- All scenario-derived metrics in this report are proxy metrics unless explicitly marked manual-review-labeled.",
        "- `scenario_canonical == cough` is treated as cough-session proxy positive, not strict ground truth.",
        "",
        "## p01 / p02 / p03 Cough-Session Activation Rate",
        "",
        f"- p01 @0.50: v3.1 `{findings['p01_cough_session_activation_rate']['threshold_0_50']['v3_1']['activation_rate']}` vs v3.2 `{findings['p01_cough_session_activation_rate']['threshold_0_50']['v3_2']['activation_rate']}`",
        f"- p02 @0.50: v3.1 `{findings['p02_cough_session_activation_rate']['threshold_0_50']['v3_1']['activation_rate']}` vs v3.2 `{findings['p02_cough_session_activation_rate']['threshold_0_50']['v3_2']['activation_rate']}`",
        f"- p03 @0.50: v3.1 `{findings['p03_cough_session_activation_rate']['threshold_0_50']['v3_1']['activation_rate']}` vs v3.2 `{findings['p03_cough_session_activation_rate']['threshold_0_50']['v3_2']['activation_rate']}`",
        "",
        "## long_quiet_eval Event-Level FP/hour",
        "",
        f"- v3.1 @0.50 event FP/hour: `{None if long_quiet_v31 is None else long_quiet_v31['event_level_fp_per_hour']}``",
        f"- v3.2 @0.50 event FP/hour: `{None if long_quiet_v32 is None else long_quiet_v32['event_level_fp_per_hour']}``",
        "",
        "## FP Source Ranking @0.50",
        "",
    ]
    for row in report["fp_source_ranking_threshold_0_50"][:8]:
        lines.append(
            f"- `{row['scenario_canonical']}` rows `{row['rows']}`, v3.1 FP `{row['v3_1_fp']}`, v3.2 FP `{row['v3_2_fp']}`, delta `{row['delta_fp']}`"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        *[f"- {reason}" for reason in recommendation["reasons"]],
        *[f"- sampler direction: {item}" for item in recommendation["sampler_adjustment_direction"]],
        "",
        "## Not Executed",
        "",
        *[f"- {item}" for item in report["not_executed"]],
        "",
        "## Known-Error Regression",
        "",
        "- This stage does not treat known-error regression as pass unless a ready manifest is available and scored.",
        f"- current status: `{report['known_error_regression']['status']}`",
    ])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = path_arg(args.config)
    checkpoint_path = path_arg(args.checkpoint)
    manifest_path = path_arg(args.manifest)
    v3_1_scored_path = path_arg(args.v3_1_scored)
    report_path = path_arg(args.report)
    summary_path = path_arg(args.summary_md)
    threshold_sweep_path = path_arg(args.threshold_sweep_csv)
    comparison_path = path_arg(args.comparison_csv)

    for path in (report_path, summary_path, threshold_sweep_path, comparison_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {path}")

    config = load_config(config_path)
    frontend = frontend_summary(config)
    if frontend["status"] != "passed":
        raise ValueError(f"board_htk_no_norm_v1 config validation failed: {frontend['checks']}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    manifest = read_csv(manifest_path)
    manifest_validation = validate_manifest(manifest)
    joined_input, join_report = attach_v3_1_scores(manifest, read_csv(v3_1_scored_path))
    if not all(join_report["strict_validation"].values()):
        raise ValueError(f"Strict join validation failed: {join_report}")

    joined_input = build_truth_columns(joined_input)
    scored, score_report = score_v3_2(joined_input, config, checkpoint_path)

    scored_rows = int(len(scored))
    strict_scored_rows_ok = scored_rows == int(len(manifest))
    missing_v3_2_score_rows = int(pd.to_numeric(scored["v3_2_cough_prob"], errors="coerce").isna().sum())
    if not strict_scored_rows_ok or missing_v3_2_score_rows != 0:
        raise ValueError(
            f"Strict scoring validation failed: scored_rows={scored_rows}, manifest_rows={len(manifest)}, missing_v3_2_score_rows={missing_v3_2_score_rows}"
        )

    threshold_sweep = build_threshold_sweep(scored)
    fp_ranking = rank_fp_sources(scored, 0.50)
    proxy_findings = compute_proxy_findings(scored)
    recommendation = derive_recommendation(scored, fp_ranking, proxy_findings)

    known_error_status = {"status": "not_completed", "reason": "no ready manifest was supplied in this task input set"}

    report = {
        "status": "passed",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "v3_1_scored_manifest": str(v3_1_scored_path),
        "resolved_model_name": score_report["resolved_model_name"],
        "frontend": frontend,
        "class_order": score_report["class_order"],
        "expected_input_shape": score_report["expected_input_shape"],
        "observed_input_shapes": [list(shape) for shape in score_report["observed_input_shapes"]],
        "rows": {
            "manifest": int(len(manifest)),
            "scored": scored_rows,
            "joined_v3_1": join_report["joined_rows"],
        },
        "manifest_validation": manifest_validation,
        "join_validation": {
            **join_report,
            "v3_2_scored_rows": scored_rows,
            "missing_v3_2_score_rows": missing_v3_2_score_rows,
            "strict_scored_rows_eq_manifest": strict_scored_rows_ok,
        },
        "audio": {
            "path_resolution_counts": score_report["audio_path_resolution_counts"],
            "files_loaded": score_report["audio_files_loaded"],
            "source_policy": score_report["audio_source_policy"],
            "channel_handling": "16 kHz mono via librosa.load(..., mono=True)",
        },
        "proxy_findings": proxy_findings,
        "comparison_by_scenario": comparison_summary(scored, "scenario_canonical"),
        "comparison_by_split_role": comparison_summary(scored, "split_role_suggested"),
        "comparison_by_person": comparison_summary(scored, "person_id_for_analysis"),
        "fp_source_ranking_threshold_0_50": fp_ranking,
        "known_error_regression": known_error_status,
        "recommendation": recommendation,
        "not_executed": NOT_EXECUTED,
        "proxy_metric_note": "Scenario-hint outputs are proxy metrics only and are not deployment threshold conclusions.",
        "manual_review_metric_note": "Manual-review-labeled event metrics are not available in this stage unless a ready labeled manifest is supplied.",
    }

    public_scored = scored.drop(columns=["_join_key"], errors="ignore")
    ensure_dir(comparison_path.parent)
    public_scored.to_csv(comparison_path, index=False)
    ensure_dir(threshold_sweep_path.parent)
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    write_json(report_path, report)
    write_summary(report, summary_path)

    print(
        json.dumps(
            {
                "status": "passed",
                "comparison_csv": str(comparison_path),
                "threshold_sweep_csv": str(threshold_sweep_path),
                "report_json": str(report_path),
                "summary_md": str(summary_path),
                "rows": scored_rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
