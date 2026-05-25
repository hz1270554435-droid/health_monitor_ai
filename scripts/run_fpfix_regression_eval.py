from __future__ import annotations

import argparse
import json
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


THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
EXCLUDED_DECISIONS = {"uncertain", "exclude"}
NEGATIVE_DECISIONS = {"hard_negative", "clean_non_cough"}
P0_BUCKETS = ["P0_hard_negative_speech", "P0_hard_negative_room_or_silence"]
REQUIRED_BUCKETS = [
    "P0_hard_negative_speech",
    "P0_hard_negative_room_or_silence",
    "P1_possible_true_cough",
    "P1_possible_v3_false_negative",
    "P1_high_v3_other_yamnet",
    "P2_mid_confusion",
    "P3_random_background",
]
NOT_EXECUTED = [
    "training",
    "model export",
    "board deployment",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
]
AUDIO_SOURCE_FIELDS = ["original_path", "source_audio", "audio_file"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run board_live_data_1 FP-fix known-error regression eval.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--review-sheet", default="review/board_live_data_1/review_sheet_cleaned.csv")
    parser.add_argument("--v3-scored", default="review/board_live_data_1/v3_scored_manifest.csv")
    parser.add_argument("--checkpoint", default="models/audio/audio_baseline_v3_1_fpfix/best_model.pt")
    parser.add_argument("--out", default="review/board_live_data_1/fpfix_regression_scored.csv")
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_regression_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_regression_summary.md")
    parser.add_argument("--expected-eligible-rows", type=int, default=317)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def path_arg(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
        "mono": True,
        "window_seconds": float(audio.get("window_seconds")) == 1.0,
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
            "hop_sec": float(audio.get("hop_seconds", 0.5)),
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
        raise RuntimeError("torch is required for FP-fix regression eval") from exc
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return frame


def prepare_review_rows(review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "manual_decision" not in review.columns:
        raise ValueError("review sheet missing manual_decision")
    decisions = review["manual_decision"].astype(str).str.strip().str.lower()
    excluded = review[decisions.isin(EXCLUDED_DECISIONS)].copy()
    eligible = review[~decisions.isin(EXCLUDED_DECISIONS)].copy()
    eligible["manual_decision"] = eligible["manual_decision"].astype(str).str.strip()
    eligible["_join_key"] = eligible.apply(normalized_join_key, axis=1)
    excluded["_join_key"] = excluded.apply(normalized_join_key, axis=1)
    duplicate_count = int(eligible.duplicated("_join_key").sum())
    if duplicate_count:
        raise ValueError(f"Eligible review rows contain {duplicate_count} duplicate normalized keys")
    return eligible.reset_index(drop=True), excluded.reset_index(drop=True)


def attach_v3_scores(eligible: pd.DataFrame, v3_scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    v3 = v3_scored.copy()
    required = ["original_path", "window_start_sec", "window_end_sec", "v3_cough_prob"]
    missing = [name for name in required if name not in v3.columns]
    if missing:
        raise ValueError(f"v3 scored manifest missing columns: {missing}")
    v3["_join_key"] = v3.apply(normalized_join_key, axis=1)
    duplicate_v3 = int(v3.duplicated("_join_key").sum())
    if duplicate_v3:
        raise ValueError(f"v3 scored manifest contains {duplicate_v3} duplicate normalized keys")

    v3_columns = ["_join_key", "v3_cough_prob"]
    for column in ("v3_pred", "v3_non_cough_prob", "v3_score_bucket"):
        if column in v3.columns:
            v3_columns.append(column)
    joined = eligible.merge(
        v3[v3_columns].rename(
            columns={
                "v3_cough_prob": "_v3_manifest_cough_prob",
                "v3_pred": "_v3_manifest_pred",
                "v3_non_cough_prob": "_v3_manifest_non_cough_prob",
                "v3_score_bucket": "_v3_manifest_score_bucket",
            }
        ),
        on="_join_key",
        how="left",
    )
    missing_matches = int(joined["_v3_manifest_cough_prob"].isna().sum())
    if missing_matches:
        raise ValueError(f"Failed to match {missing_matches} eligible rows to v3_scored manifest")

    existing_v3 = pd.to_numeric(joined.get("v3_cough_prob"), errors="coerce") if "v3_cough_prob" in joined else None
    manifest_v3 = pd.to_numeric(joined["_v3_manifest_cough_prob"], errors="coerce")
    mismatch_count = 0
    max_abs_diff = 0.0
    if existing_v3 is not None:
        diffs = (existing_v3 - manifest_v3).abs().dropna()
        mismatch_count = int((diffs > 1e-6).sum())
        max_abs_diff = float(diffs.max()) if not diffs.empty else 0.0

    joined["v3_cough_prob"] = manifest_v3
    if "_v3_manifest_non_cough_prob" in joined.columns:
        joined["v3_non_cough_prob"] = pd.to_numeric(joined["_v3_manifest_non_cough_prob"], errors="coerce")
    if "_v3_manifest_pred" in joined.columns:
        joined["v3_pred"] = joined["_v3_manifest_pred"]
    if "_v3_manifest_score_bucket" in joined.columns:
        joined["v3_score_bucket"] = joined["_v3_manifest_score_bucket"]

    joined = joined.drop(columns=[name for name in joined.columns if name.startswith("_v3_manifest_")])
    return joined, {
        "v3_rows": int(len(v3)),
        "eligible_matches": int(len(joined)),
        "duplicate_v3_keys": duplicate_v3,
        "existing_v3_score_mismatch_count": mismatch_count,
        "existing_v3_score_max_abs_diff": max_abs_diff,
        "join_key_normalization": {
            "original_path": "normcase(normpath(path))",
            "window_start_sec": "round_to_3_decimals",
            "window_end_sec": "round_to_3_decimals",
        },
    }


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
        f"clip_path is intentionally ignored for row key={row.get('_join_key')}"
    )


def score_fpfix(
    rows: pd.DataFrame,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, model_name = load_checkpoint(torch, checkpoint_path, config, device)
    if class_names != ["non_cough", "cough"]:
        raise ValueError(f"Unexpected class_names: {class_names}")
    expected_shape = expected_input_shape(config["audio"])
    cough_index = int(class_names.index("cough"))
    sample_rate = int(config["audio"]["sample_rate"])
    window_size = int(round(sample_rate * float(config["audio"]["window_seconds"])))
    audio_cache: dict[str, np.ndarray] = {}
    audio_source_counts = {field: 0 for field in AUDIO_SOURCE_FIELDS}
    observed_shapes: list[list[int]] = []
    cough_probs: list[float] = []
    preds: list[str] = []
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
        cough_prob = float(probs[cough_index])
        cough_probs.append(cough_prob)
        preds.append("cough" if cough_prob >= 0.50 else "non_cough")
        observed_shapes.append(observed_shape)
        audio_paths.append(str(audio_path))
        audio_source_fields.append(source_field)

    output = rows.copy()
    output["regression_audio_source_field"] = audio_source_fields
    output["regression_audio_path"] = audio_paths
    output["fpfix_cough_prob"] = cough_probs
    output["fpfix_pred_at_0_50"] = preds
    output["delta_cough_prob"] = pd.to_numeric(output["fpfix_cough_prob"]) - pd.to_numeric(output["v3_cough_prob"])
    return output, {
        "resolved_model_name": model_name,
        "class_order": class_names,
        "expected_input_shape": expected_shape,
        "observed_input_shapes": sorted({tuple(shape) for shape in observed_shapes}),
        "device_used": str(device),
        "audio_path_resolution_counts": audio_source_counts,
        "audio_files_loaded": int(len(audio_cache)),
        "audio_source_policy": "original_path -> source_audio -> audio_file; clip_path ignored",
    }


def confusion_counts(frame: pd.DataFrame, score_column: str, threshold: float) -> dict[str, Any]:
    y_true = frame["manual_decision"].eq("cough").astype(int)
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
    }


def threshold_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        v3 = confusion_counts(frame, "v3_cough_prob", threshold)
        fpfix = confusion_counts(frame, "fpfix_cough_prob", threshold)
        rows.append(
            {
                "threshold": threshold,
                "v3": v3,
                "fpfix": fpfix,
                "delta": {
                    "tp": fpfix["tp"] - v3["tp"],
                    "fp": fpfix["fp"] - v3["fp"],
                    "tn": fpfix["tn"] - v3["tn"],
                    "fn": fpfix["fn"] - v3["fn"],
                    "precision": fpfix["precision"] - v3["precision"],
                    "recall": fpfix["recall"] - v3["recall"],
                    "f1": fpfix["f1"] - v3["f1"],
                },
            }
        )
    return rows


def grouped_error_table(frame: pd.DataFrame, group_column: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    groups = sorted(str(value) for value in frame[group_column].dropna().astype(str).unique())
    if group_column == "review_bucket":
        groups = REQUIRED_BUCKETS + [group for group in groups if group not in REQUIRED_BUCKETS]
    for threshold in THRESHOLDS:
        rows: list[dict[str, Any]] = []
        for group in groups:
            subset = frame[frame[group_column].astype(str).eq(group)]
            if subset.empty:
                rows.append({"group": group, "rows": 0, "v3_fp": 0, "v3_fn": 0, "fpfix_fp": 0, "fpfix_fn": 0})
                continue
            v3 = confusion_counts(subset, "v3_cough_prob", threshold)
            fpfix = confusion_counts(subset, "fpfix_cough_prob", threshold)
            rows.append(
                {
                    "group": group,
                    "rows": int(len(subset)),
                    "v3_fp": v3["fp"],
                    "v3_fn": v3["fn"],
                    "fpfix_fp": fpfix["fp"],
                    "fpfix_fn": fpfix["fn"],
                    "fp_delta": fpfix["fp"] - v3["fp"],
                    "fn_delta": fpfix["fn"] - v3["fn"],
                }
            )
        output[f"{threshold:.2f}"] = rows
    return output


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


def distribution_by_manual_decision(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for decision, subset in frame.groupby("manual_decision"):
        output[str(decision)] = {
            "rows": int(len(subset)),
            "v3_cough_prob": quantiles(subset["v3_cough_prob"]),
            "fpfix_cough_prob": quantiles(subset["fpfix_cough_prob"]),
            "delta_cough_prob": quantiles(subset["delta_cough_prob"]),
        }
    return output


def p0_probability_quantiles(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for bucket in P0_BUCKETS:
        subset = frame[frame["review_bucket"].astype(str).eq(bucket)]
        output[bucket] = {
            "rows": int(len(subset)),
            "v3_cough_prob": quantiles(subset["v3_cough_prob"]),
            "fpfix_cough_prob": quantiles(subset["fpfix_cough_prob"]),
            "delta_cough_prob": quantiles(subset["delta_cough_prob"], include_min_max=False),
        }
    return output


def threshold_lookup(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    key = round(threshold, 2)
    for row in rows:
        if round(float(row["threshold"]), 2) == key:
            return row
    raise KeyError(threshold)


def focus_checks(frame: pd.DataFrame, thresholds: list[dict[str, Any]]) -> dict[str, Any]:
    at_090 = threshold_lookup(thresholds, 0.90)
    p0 = frame[frame["review_bucket"].isin(P0_BUCKETS)]
    p0_v3 = confusion_counts(p0, "v3_cough_prob", 0.90)
    p0_fpfix = confusion_counts(p0, "fpfix_cough_prob", 0.90)
    p0_speech = frame[frame["review_bucket"].astype(str).eq("P0_hard_negative_speech")]
    p0_room = frame[frame["review_bucket"].astype(str).eq("P0_hard_negative_room_or_silence")]
    p1_fn = frame[frame["review_bucket"].astype(str).eq("P1_possible_v3_false_negative")]

    recall_drop: dict[str, float] = {}
    recall_drop_aliases: dict[str, float] = {}
    for threshold in (0.50, 0.70, 0.90):
        row = threshold_lookup(thresholds, threshold)
        value = row["v3"]["recall"] - row["fpfix"]["recall"]
        recall_drop[f"cough_recall_drop_at_{threshold:.2f}"] = value
        recall_drop_aliases[f"cough_recall_drop_at_{threshold:.2f}".replace(".", "_")] = value

    p0_improved = p0_fpfix["fp"] < p0_v3["fp"]
    guardrail_ok = (
        recall_drop_aliases["cough_recall_drop_at_0_50"] <= 0.10
        and recall_drop_aliases["cough_recall_drop_at_0_70"] <= 0.15
    )
    if p0_improved and guardrail_ok:
        recommendation = "live_smoke"
    elif p0_improved:
        recommendation = "inspect_failures"
    else:
        recommendation = "revise_training"

    return {
        "threshold_0_90": {
            "v3_fp": at_090["v3"]["fp"],
            "fpfix_fp": at_090["fpfix"]["fp"],
            "fp_delta": at_090["delta"]["fp"],
        },
        "p0_combined_at_0_90": {
            "v3_fp": p0_v3["fp"],
            "fpfix_fp": p0_fpfix["fp"],
            "fp_delta": p0_fpfix["fp"] - p0_v3["fp"],
        },
        "p0_hard_negative_speech_at_0_90": {
            "v3_fp": confusion_counts(p0_speech, "v3_cough_prob", 0.90)["fp"],
            "fpfix_fp": confusion_counts(p0_speech, "fpfix_cough_prob", 0.90)["fp"],
        },
        "p0_hard_negative_room_or_silence_at_0_90": {
            "v3_fp": confusion_counts(p0_room, "v3_cough_prob", 0.90)["fp"],
            "fpfix_fp": confusion_counts(p0_room, "fpfix_cough_prob", 0.90)["fp"],
        },
        "p1_possible_v3_false_negative_recall": {
            "threshold_0_50": {
                "v3": confusion_counts(p1_fn, "v3_cough_prob", 0.50)["recall"],
                "fpfix": confusion_counts(p1_fn, "fpfix_cough_prob", 0.50)["recall"],
            },
            "threshold_0_90": {
                "v3": confusion_counts(p1_fn, "v3_cough_prob", 0.90)["recall"],
                "fpfix": confusion_counts(p1_fn, "fpfix_cough_prob", 0.90)["recall"],
            },
        },
        **recall_drop,
        **recall_drop_aliases,
        "recommended_next_stage": recommendation,
        "recommendation_rule": (
            "live_smoke if combined P0 FP at 0.90 decreases and cough recall drop at "
            "0.50 <= 0.10 and at 0.70 <= 0.15; inspect_failures if P0 improves but "
            "guardrail fails; otherwise revise_training"
        ),
    }


def validate_scored(frame: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    errors: list[str] = []
    if len(frame) != expected_rows:
        errors.append(f"row_count={len(frame)}, expected={expected_rows}")
    unique_keys = int(frame["_join_key"].nunique())
    if unique_keys != len(frame):
        errors.append(f"unique normalized key count={unique_keys}, rows={len(frame)}")
    for column in ("v3_cough_prob", "fpfix_cough_prob", "delta_cough_prob"):
        if column not in frame.columns:
            errors.append(f"missing column {column}")
            continue
        missing = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
        if missing:
            errors.append(f"{column} has {missing} missing/non-numeric rows")
    fpfix = pd.to_numeric(frame["fpfix_cough_prob"], errors="coerce")
    out_of_range = int((~fpfix.between(0.0, 1.0)).sum())
    if out_of_range:
        errors.append(f"fpfix_cough_prob has {out_of_range} values outside [0, 1]")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "row_count": int(len(frame)),
        "unique_normalized_key_count": unique_keys,
        "duplicate_normalized_key_count": int(len(frame) - unique_keys),
        "fpfix_probability_range": {
            "min": float(fpfix.min()),
            "max": float(fpfix.max()),
        },
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    focus = report["focus_checks"]
    lines = [
        "# v3.1 FP-fix Known-Error Regression Eval",
        "",
        f"- status: `{report['status']}`",
        f"- scored_rows: `{report['rows']['eligible']}`",
        f"- excluded_rows: `{report['rows']['excluded']}`",
        f"- model: `{report['resolved_model_name']}`",
        f"- checkpoint: `{report['checkpoint']}`",
        f"- recommended_next_stage: `{focus['recommended_next_stage']}`",
        "",
        "## Threshold 0.90 Focus",
        "",
        f"- v3 FP: `{focus['threshold_0_90']['v3_fp']}`",
        f"- fpfix FP: `{focus['threshold_0_90']['fpfix_fp']}`",
        f"- combined P0 v3 FP: `{focus['p0_combined_at_0_90']['v3_fp']}`",
        f"- combined P0 fpfix FP: `{focus['p0_combined_at_0_90']['fpfix_fp']}`",
        f"- cough_recall_drop_at_0_50: `{focus['cough_recall_drop_at_0_50']}`",
        f"- cough_recall_drop_at_0_70: `{focus['cough_recall_drop_at_0_70']}`",
        f"- cough_recall_drop_at_0_90: `{focus['cough_recall_drop_at_0_90']}`",
        "",
        "## Threshold Comparison",
        "",
    ]
    for row in report["threshold_comparison"]:
        lines.append(
            f"- `{row['threshold']:.2f}` v3 FP/FN `{row['v3']['fp']}/{row['v3']['fn']}`, "
            f"fpfix FP/FN `{row['fpfix']['fp']}/{row['fpfix']['fn']}`, "
            f"v3 recall `{row['v3']['recall']:.4f}`, fpfix recall `{row['fpfix']['recall']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## P0 Probability Quantiles",
            "",
            f"`{report['p0_probability_quantiles']}`",
            "",
            "## Notes",
            "",
            "- board_live_data_1 entered FP-fix training; this is a known-error regression check, not a generalization evaluation.",
            "- Review clips are not used as evaluation source; original 1.0s source windows are used.",
            "",
            "## Not Executed",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in report["not_executed"]])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = path_arg(args.config)
    review_path = path_arg(args.review_sheet)
    v3_path = path_arg(args.v3_scored)
    checkpoint_path = path_arg(args.checkpoint)
    out_path = path_arg(args.out)
    report_path = path_arg(args.report)
    summary_path = path_arg(args.summary_md)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {report_path}")

    config = load_config(config_path)
    frontend = frontend_summary(config)
    if frontend["status"] != "passed":
        raise ValueError(f"board_htk_no_norm_v1 config validation failed: {frontend['checks']}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    review = read_csv(review_path)
    v3_scored = read_csv(v3_path)
    eligible, excluded = prepare_review_rows(review)
    eligible, join_report = attach_v3_scores(eligible, v3_scored)
    scored, score_report = score_fpfix(eligible, config, checkpoint_path)
    validation = validate_scored(scored, args.expected_eligible_rows)
    if validation["status"] != "passed":
        raise ValueError(f"scored output validation failed: {validation['errors']}")

    threshold_comparison = threshold_table(scored)
    focus = focus_checks(scored, threshold_comparison)
    report = {
        "status": "passed",
        "review_sheet": str(review_path),
        "v3_scored_manifest": str(v3_path),
        "out_csv": str(out_path),
        "report_json": str(report_path),
        "summary_md": str(summary_path),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "resolved_model_name": score_report["resolved_model_name"],
        "frontend": frontend,
        "class_order": score_report["class_order"],
        "expected_input_shape": score_report["expected_input_shape"],
        "observed_input_shapes": [list(shape) for shape in score_report["observed_input_shapes"]],
        "rows": {
            "input": int(len(review)),
            "eligible": int(len(scored)),
            "excluded": int(len(excluded)),
        },
        "excluded_rows": excluded.drop(columns=["_join_key"], errors="ignore").to_dict(orient="records"),
        "join": join_report,
        "audio": {
            "path_resolution_counts": score_report["audio_path_resolution_counts"],
            "files_loaded": score_report["audio_files_loaded"],
            "source_policy": score_report["audio_source_policy"],
            "channel_handling": "16 kHz mono via librosa.load(..., mono=True); clip_path ignored",
        },
        "validation": validation,
        "threshold_comparison": threshold_comparison,
        "per_bucket_error_comparison": grouped_error_table(scored, "review_bucket"),
        "per_manual_subtype_norm_error_comparison": grouped_error_table(scored, "manual_subtype_norm"),
        "probability_distribution_by_manual_decision": distribution_by_manual_decision(scored),
        "p0_probability_quantiles": p0_probability_quantiles(scored),
        "focus_checks": focus,
        "recommended_next_stage": focus["recommended_next_stage"],
        "known_error_regression_only": True,
        "generalization_claim": False,
        "generalization_note": "board_live_data_1 entered training; this is not a held-out generalization eval set.",
        "not_executed": NOT_EXECUTED,
    }

    public_scored = scored.drop(columns=["_join_key"], errors="ignore")
    ensure_dir(out_path.parent)
    public_scored.to_csv(out_path, index=False)
    write_json(report, report_path)
    write_summary(report, summary_path)
    print(json.dumps({"status": "passed", "rows": int(len(public_scored)), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
