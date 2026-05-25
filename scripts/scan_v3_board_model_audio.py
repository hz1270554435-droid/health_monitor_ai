from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


REQUIRED_NONEMPTY_COLUMNS = [
    "original_path",
    "session_id",
    "person_id",
    "window_start_sec",
    "window_end_sec",
]
UNIQUE_KEY_COLUMNS = ["original_path", "window_start_sec", "window_end_sec"]
V3_SCORE_COLUMNS = [
    "v3_cough_prob",
    "v3_non_cough_prob",
    "v3_pred",
    "v3_score_bucket",
]
VALID_BUCKETS = {
    "v3_high_ge_0_90",
    "v3_mid_0_50_0_90",
    "v3_low_lt_0_50",
}
AUDIO_CHANNEL_HANDLING_MODE = (
    "16k_mono_via_v3_1_mining_common.load_audio_mono; "
    "WAV uses librosa.load(..., mono=True), PCM multi-channel uses mean(axis=1)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score v3.1 mining windows with the deployed v3 board model."
    )
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-rows", type=int, default=4090)
    parser.add_argument("--prob-epsilon", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_torch() -> tuple[Any, dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to score the v3 board model") from exc

    env = {
        "import_ok": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    return torch, env


def as_float(row: pd.Series, name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {name}={row.get(name, '')!r}") from exc


def expected_input_shape(audio_config: dict[str, Any]) -> list[int]:
    sample_rate = int(audio_config["sample_rate"])
    window_samples = int(round(sample_rate * float(audio_config["window_seconds"])))
    hop_length = int(audio_config.get("hop_length", 160))
    n_fft = int(audio_config.get("n_fft", 1024))
    n_mels = int(audio_config.get("n_mels", 40))
    center = bool(audio_config.get("center", True))

    if center:
        time_bins = 1 + (window_samples // hop_length)
    else:
        time_bins = 1 + max(0, (window_samples - n_fft) // hop_length)
    return [1, 1, n_mels, time_bins]


def validate_model_config(model_config: dict[str, Any]) -> None:
    audio_config = model_config["audio"]
    labels_config = model_config["labels"]
    class_order = list(labels_config.get("class_order", []))
    if str(audio_config.get("frontend", "")) != "board_htk_no_norm_v1":
        raise ValueError(f"Unexpected frontend: {audio_config.get('frontend')}")
    if str(audio_config.get("normalize", "")).lower() != "none":
        raise ValueError(f"Unexpected normalization: {audio_config.get('normalize')}")
    if class_order != ["non_cough", "cough"]:
        raise ValueError(f"Unexpected class_order: {class_order}")
    if expected_input_shape(audio_config) != [1, 1, 40, 101]:
        raise ValueError(
            f"Unexpected expected input shape from config: {expected_input_shape(audio_config)}"
        )


def load_checkpoint(torch, checkpoint_path: Path, config: dict[str, Any], device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    labels_cfg = config["labels"]
    class_names = list(
        checkpoint.get("class_names")
        or labels_cfg.get("class_order")
        or [labels_cfg["negative"], labels_cfg["positive"]]
    )
    model_name = str(
        checkpoint.get("model_name", config.get("training", {}).get("model", "small_cnn"))
    )
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, model_name


def validate_input_manifest(manifest: pd.DataFrame, expected_rows: int | None) -> dict[str, int]:
    missing = [name for name in REQUIRED_NONEMPTY_COLUMNS if name not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    if "manual_decision" in manifest.columns:
        raise ValueError("manual_decision must not be present for weak scoring")

    for name in REQUIRED_NONEMPTY_COLUMNS:
        empty_count = int((manifest[name].astype(str).str.strip() == "").sum())
        if empty_count:
            raise ValueError(f"Manifest column {name} has {empty_count} empty rows")

    unique_key_count = int(manifest.drop_duplicates(subset=UNIQUE_KEY_COLUMNS).shape[0])
    row_count = int(len(manifest))
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"Manifest row_count={row_count}, expected_rows={expected_rows}")
    if expected_rows is not None and unique_key_count != expected_rows:
        raise ValueError(
            f"Manifest unique key count={unique_key_count}, expected_rows={expected_rows}"
        )
    if unique_key_count != row_count:
        raise ValueError(f"Manifest has duplicate window keys: {row_count - unique_key_count}")

    return {
        "row_count": row_count,
        "unique_key_count": unique_key_count,
        "duplicate_key_count": row_count - unique_key_count,
    }


def bucket_for(cough_prob: float) -> str:
    if cough_prob >= 0.90:
        return "v3_high_ge_0_90"
    if cough_prob >= 0.50:
        return "v3_mid_0_50_0_90"
    return "v3_low_lt_0_50"


def score_manifest_row(
    row: pd.Series,
    row_index: int,
    torch,
    model,
    device,
    audio_config: dict[str, Any],
    pcm_defaults: dict[str, Any],
    class_names: list[str],
    cough_index: int,
    non_cough_index: int,
    expected_shape: list[int],
    audio_cache: dict[str, Any],
) -> tuple[dict[str, object], list[int]]:
    audio_ref = str(row.get("audio_file", row.get("original_path", "")))
    if not audio_ref:
        raise ValueError(f"Row {row_index} has no audio_file/original_path")

    sample_rate = int(audio_config["sample_rate"])
    audio = audio_cache.get(audio_ref)
    if audio is None:
        audio = load_audio_mono(resolve_path(audio_ref), sample_rate, pcm_defaults)
        audio_cache[audio_ref] = audio

    start_time = as_float(row, "window_start_sec")
    end_time = as_float(row, "window_end_sec")
    window_size = int(round(sample_rate * float(audio_config["window_seconds"])))
    start_sample = max(0, int(round(start_time * sample_rate)))
    end_sample = int(round(end_time * sample_rate))
    if end_sample <= start_sample:
        end_sample = start_sample + window_size

    chunk = pad_window(audio[start_sample:end_sample], window_size)
    feature = log_mel_feature(chunk, audio_config)
    tensor = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
    observed_shape = [int(value) for value in tensor.shape]
    if observed_shape != expected_shape:
        raise ValueError(
            f"Row {row_index} produced input shape {observed_shape}, expected {expected_shape}"
        )

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    cough_prob = float(probs[cough_index])
    non_cough_prob = float(probs[non_cough_index])
    pred = class_names[int(probs.argmax())]
    output = row.to_dict()
    output.update(
        {
            "v3_cough_prob": round(cough_prob, 9),
            "v3_non_cough_prob": round(non_cough_prob, 9),
            "v3_pred": pred,
            "v3_score_bucket": bucket_for(cough_prob),
        }
    )
    return output, observed_shape


def prob_stats(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "min": float(numeric.min()),
        "mean": float(numeric.mean()),
        "p50": float(numeric.quantile(0.50)),
        "p90": float(numeric.quantile(0.90)),
        "p95": float(numeric.quantile(0.95)),
        "p99": float(numeric.quantile(0.99)),
        "max": float(numeric.max()),
    }


def distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def session_duration_checks(scored: pd.DataFrame) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    counts = scored.groupby("session_id").size()
    for session_id, window_count in counts[counts == 44].sort_index().items():
        subset = scored[scored["session_id"] == session_id]
        metadata_path = Path(str(subset["metadata_json"].iloc[0]))
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checks.append(
            {
                "session_id": str(session_id),
                "window_count": int(window_count),
                "metadata_json": str(metadata_path),
                "metadata_duration_s": metadata.get("duration_s"),
            }
        )
    return checks


def validate_scored_manifest(
    scored: pd.DataFrame,
    expected_rows: int | None,
    prob_epsilon: float,
) -> dict[str, object]:
    input_validation = validate_input_manifest(scored, expected_rows)
    if "manual_decision" in scored.columns:
        raise ValueError("manual_decision must not be present in scored output")

    for name in V3_SCORE_COLUMNS:
        if name not in scored.columns:
            raise ValueError(f"Scored output missing {name}")

    sums = (
        pd.to_numeric(scored["v3_cough_prob"], errors="coerce")
        + pd.to_numeric(scored["v3_non_cough_prob"], errors="coerce")
    )
    max_prob_sum_abs_error = float((sums - 1.0).abs().max())
    if max_prob_sum_abs_error >= prob_epsilon:
        raise ValueError(
            f"Probability sanity failed: max abs sum error {max_prob_sum_abs_error}"
        )

    pred_values = set(scored["v3_pred"].astype(str).unique())
    if not pred_values <= {"non_cough", "cough"}:
        raise ValueError(f"Unexpected v3_pred values: {sorted(pred_values)}")

    bucket_values = set(scored["v3_score_bucket"].astype(str).unique())
    if not bucket_values <= VALID_BUCKETS:
        raise ValueError(f"Unexpected v3_score_bucket values: {sorted(bucket_values)}")

    return {
        **input_validation,
        "manual_decision_present": False,
        "max_probability_sum_abs_error": max_prob_sum_abs_error,
        "probability_epsilon": float(prob_epsilon),
        "probability_sanity_ok": True,
        "v3_pred_values": sorted(pred_values),
        "v3_score_bucket_values": sorted(bucket_values),
    }


def build_report(
    scored: pd.DataFrame,
    validation: dict[str, object],
    manifest_path: Path,
    out_path: Path,
    report_path: Path,
    summary_path: Path,
    model_config_path: Path,
    checkpoint_path: Path,
    model_config: dict[str, Any],
    model_name: str,
    class_names: list[str],
    expected_shape: list[int],
    observed_shape: list[int],
    torch_env: dict[str, object],
    device_name: str,
    device_used: str,
) -> dict[str, object]:
    high = scored[scored["v3_score_bucket"] == "v3_high_ge_0_90"]
    low = scored[scored["v3_score_bucket"] == "v3_low_lt_0_50"]
    silence_inside = high[high["yamnet_top1"].isin(["Silence", "Inside, small room"])]
    low_yamnet_cough_like = low[low["yamnet_top1"].isin(["Cough", "Snort", "Breathing"])]

    torch_env = dict(torch_env)
    torch_env.update({"device_requested": device_name, "device_used": device_used})

    return {
        "manifest_csv": str(manifest_path),
        "out_csv": str(out_path),
        "report_json": str(report_path),
        "summary_md": str(summary_path),
        "resolved_model_config": str(model_config_path),
        "resolved_checkpoint": str(checkpoint_path),
        "frontend": str(model_config["audio"].get("frontend")),
        "class_order": list(model_config["labels"].get("class_order", class_names)),
        "checkpoint_class_names": list(class_names),
        "model_name": model_name,
        "expected_model_input_shape": expected_shape,
        "observed_scorer_input_shape": observed_shape,
        "audio_channel_handling_mode": AUDIO_CHANNEL_HANDLING_MODE,
        "torch_env": torch_env,
        "validation": validation,
        "total_windows": int(len(scored)),
        "v3_cough_prob_summary": prob_stats(scored["v3_cough_prob"]),
        "v3_score_bucket_counts": distribution(scored["v3_score_bucket"]),
        "v3_high_ge_0_90_count": int(len(high)),
        "v3_high_ge_0_90_yamnet_top1_distribution": distribution(high["yamnet_top1"]),
        "v3_high_ge_0_90_yamnet_top1_speech_count": int((high["yamnet_top1"] == "Speech").sum()),
        "v3_high_ge_0_90_yamnet_top1_silence_or_inside_small_room_count": int(
            len(silence_inside)
        ),
        "v3_low_lt_0_50_yamnet_top1_cough_snort_breathing_count": int(
            len(low_yamnet_cough_like)
        ),
        "person_id_distribution": distribution(scored["person_id"]),
        "session_id_distribution": distribution(scored["session_id"]),
        "sessions_with_44_windows": session_duration_checks(scored),
        "notes": [
            "weak scoring only",
            "no manual_decision",
            "no review clips",
            "no candidate labels",
            "no training",
            "no raw audio copy",
            "Respiratory sounds semantics unchanged",
        ],
    }


def write_summary_md(path: Path, report: dict[str, Any]) -> None:
    prob = report["v3_cough_prob_summary"]
    lines = [
        "# v3 board model scoring summary",
        "",
        f"- manifest: `{report['manifest_csv']}`",
        f"- scored_manifest: `{report['out_csv']}`",
        f"- total_windows: `{report['total_windows']}`",
        f"- model_config: `{report['resolved_model_config']}`",
        f"- checkpoint: `{report['resolved_checkpoint']}`",
        f"- frontend: `{report['frontend']}`",
        f"- class_order: `{','.join(report['class_order'])}`",
        f"- input_shape: `{report['observed_scorer_input_shape']}`",
        f"- audio_channel_handling_mode: `{report['audio_channel_handling_mode']}`",
        "",
        "## v3 cough probability",
        "",
        (
            f"- min={prob['min']:.6f}, mean={prob['mean']:.6f}, "
            f"p50={prob['p50']:.6f}, p90={prob['p90']:.6f}, "
            f"p95={prob['p95']:.6f}, p99={prob['p99']:.6f}, max={prob['max']:.6f}"
        ),
        "",
        "## High-score checks",
        "",
        f"- v3_high_ge_0_90_count: `{report['v3_high_ge_0_90_count']}`",
        f"- high + yamnet_top1=Speech: `{report['v3_high_ge_0_90_yamnet_top1_speech_count']}`",
        (
            "- high + yamnet_top1 in {Silence, Inside, small room}: "
            f"`{report['v3_high_ge_0_90_yamnet_top1_silence_or_inside_small_room_count']}`"
        ),
        (
            "- low + yamnet_top1 in {Cough, Snort, Breathing}: "
            f"`{report['v3_low_lt_0_50_yamnet_top1_cough_snort_breathing_count']}`"
        ),
        "",
        "## High-score YAMNet top1 distribution",
        "",
    ]
    for label, count in report["v3_high_ge_0_90_yamnet_top1_distribution"].items():
        lines.append(f"- {label}: `{count}`")

    lines.extend(["", "## 44-window sessions", ""])
    for item in report["sessions_with_44_windows"]:
        lines.append(
            f"- {item['session_id']}: windows=`{item['window_count']}`, "
            f"metadata_duration_s=`{item.get('metadata_duration_s')}`"
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- row_count: `{report['validation']['row_count']}`",
            f"- unique_key_count: `{report['validation']['unique_key_count']}`",
            f"- manual_decision_present: `{report['validation']['manual_decision_present']}`",
            (
                "- max_probability_sum_abs_error: "
                f"`{report['validation']['max_probability_sum_abs_error']:.9f}`"
            ),
            "",
            "## Notes",
            "",
            "- weak scoring only",
            "- no manual_decision",
            "- no review clips",
            "- no candidate labels",
            "- no training",
            "- no raw audio copy",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.expected_rows is not None and args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive when provided")

    torch, torch_env = require_torch()
    mining_config = load_config(args.config)
    model_config_path = resolve_path(args.model_config or mining_config["model"]["config"])
    model_config = load_config(model_config_path)
    manifest_path = resolve_path(args.manifest or mining_config["paths"]["manifest_csv"])
    out_path = resolve_path(args.out or mining_config["paths"]["v3_board_scores_csv"])
    report_path = resolve_path(args.report or out_path.with_name(f"{out_path.stem}_report.json"))
    summary_path = resolve_path(args.summary_md or out_path.with_name("v3_score_summary.md"))
    checkpoint_path = resolve_path(args.checkpoint or mining_config["model"]["checkpoint"])

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite to replace: {report_path}")
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Summary exists; pass --overwrite to replace: {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"v3 checkpoint not found: {checkpoint_path}")

    validate_model_config(model_config)
    audio_config = dict(model_config["audio"])
    expected_shape = expected_input_shape(audio_config)
    class_order = list(model_config["labels"].get("class_order", []))
    preflight = {
        "resolved_model_config": str(model_config_path),
        "resolved_checkpoint": str(checkpoint_path),
        "frontend": audio_config.get("frontend"),
        "class_order": class_order,
        "expected_model_input_shape": expected_shape,
        "audio_channel_handling_mode": AUDIO_CHANNEL_HANDLING_MODE,
        "torch_env": torch_env,
    }
    print("V3 scorer preflight:")
    print(json.dumps(preflight, indent=2, ensure_ascii=False))

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    expected_rows = int(args.limit) if args.limit is not None else int(args.expected_rows)
    validate_input_manifest(manifest, expected_rows)

    training = model_config.get("training", {})
    device_name = str(training.get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)
    model, class_names, model_name = load_checkpoint(torch, checkpoint_path, model_config, device)

    positive_label = str(model_config["labels"]["positive"])
    negative_label = str(model_config["labels"]["negative"])
    if positive_label not in class_names or negative_label not in class_names:
        raise ValueError(
            f"Checkpoint classes {class_names} do not include {negative_label}/{positive_label}"
        )
    cough_index = int(class_names.index(positive_label))
    non_cough_index = int(class_names.index(negative_label))

    rows: list[dict[str, object]] = []
    observed_shape: list[int] | None = None
    audio_cache: dict[str, Any] = {}
    pcm_defaults = dict(mining_config.get("mining", {}).get("pcm_defaults", {}))
    input_columns = [column for column in manifest.columns if column not in V3_SCORE_COLUMNS]

    for row_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        scored_row, current_shape = score_manifest_row(
            row,
            row_index=row_index,
            torch=torch,
            model=model,
            device=device,
            audio_config=audio_config,
            pcm_defaults=pcm_defaults,
            class_names=class_names,
            cough_index=cough_index,
            non_cough_index=non_cough_index,
            expected_shape=expected_shape,
            audio_cache=audio_cache,
        )
        observed_shape = observed_shape or current_shape
        rows.append(scored_row)
        if row_index % 500 == 0:
            print(f"scored_rows={row_index}")

    scored = pd.DataFrame(rows, columns=input_columns + V3_SCORE_COLUMNS)
    validation = validate_scored_manifest(scored, expected_rows, float(args.prob_epsilon))
    report = build_report(
        scored=scored,
        validation=validation,
        manifest_path=manifest_path,
        out_path=out_path,
        report_path=report_path,
        summary_path=summary_path,
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        model_name=model_name,
        class_names=class_names,
        expected_shape=expected_shape,
        observed_shape=observed_shape or [],
        torch_env=torch_env,
        device_name=device_name,
        device_used=str(device),
    )

    ensure_dir(out_path.parent)
    scored.to_csv(out_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
