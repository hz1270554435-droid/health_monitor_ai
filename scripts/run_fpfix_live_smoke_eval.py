from __future__ import annotations

import argparse
import json
import re
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
from v3_1_mining_common import audio_info, iter_window_ranges, load_audio_mono, pad_window


THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
GATES = {
    "single_window": (1, 1),
    "2_of_3": (2, 3),
    "3_of_5": (3, 5),
}
NOT_EXECUTED = [
    "training",
    "model export",
    "board deployment",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
]
SCENARIO_HINTS = [
    "quiet_room",
    "normal_speech",
    "loud_speech",
    "throat_clear_laugh_breath",
    "knock_handling_noise",
    "real_cough",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fresh board-live FP-fix smoke audio on PC.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--checkpoint", default="models/audio/audio_baseline_v3_1_fpfix/best_model.pt")
    parser.add_argument("--smoke-dir", required=True)
    parser.add_argument("--out", default="review/board_live_data_1/fpfix_live_smoke_scored.csv")
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_live_smoke_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_live_smoke_summary.md")
    parser.add_argument("--top-windows-per-noncough-session", type=int, default=10)
    parser.add_argument("--top-windows-overall", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for live smoke scoring") from exc
    return torch


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


def load_checkpoint(torch, checkpoint_path: Path, config: dict[str, Any], device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    training_cfg = checkpoint_config.get("training", {}) if isinstance(checkpoint_config, dict) else {}
    model_name = str(training_cfg.get("model") or config.get("training", {}).get("model", "small_cnn")).strip().lower()
    class_names = list(
        checkpoint.get("class_names")
        or config["labels"].get("class_order")
        or [config["labels"]["negative"], config["labels"]["positive"]]
    )
    model = build_model(model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, model_name


def scenario_from_path(path: Path, smoke_dir: Path) -> str:
    try:
        parts = path.relative_to(smoke_dir).parts
    except ValueError:
        parts = path.parts
    text = " ".join(parts).lower()
    for hint in sorted(SCENARIO_HINTS, key=len, reverse=True):
        if hint in text:
            return hint
    return parts[0] if parts else "unknown"


def session_type(scenario: str) -> str:
    return "cough" if "cough" in scenario and "non_cough" not in scenario else "non_cough"


def load_metadata(audio_path: Path) -> dict[str, Any]:
    metadata_path = audio_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def session_id_from_path(audio_path: Path) -> str:
    stem = audio_path.stem
    stem = re.sub(r"_audio$", "", stem)
    return stem


def freshness_note(audio_path: Path, scenario: str) -> str:
    text = str(audio_path).lower()
    if scenario.startswith("quiet_room") and "20260424" in text:
        return "prior_quiet_background_copy"
    if "20260424" in text:
        return "prior_capture_copy"
    return "fresh_board_live_smoke"


def scan_audio_files(smoke_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in smoke_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".pcm", ".raw"}
    )
    if not paths:
        raise FileNotFoundError(f"No audio files found under smoke dir: {smoke_dir}")
    return paths


def score_audio_files(
    audio_paths: list[Path],
    smoke_dir: Path,
    config: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    torch = require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, model_name = load_checkpoint(torch, checkpoint_path, config, device)
    if class_names != ["non_cough", "cough"]:
        raise ValueError(f"Unexpected class_names: {class_names}")
    audio_config = config["audio"]
    sample_rate = int(audio_config["sample_rate"])
    window_sec = float(audio_config["window_seconds"])
    hop_sec = float(audio_config["hop_seconds"])
    window_size = int(round(sample_rate * window_sec))
    expected_shape = expected_input_shape(audio_config)
    cough_index = int(class_names.index("cough"))

    rows: list[dict[str, Any]] = []
    observed_shapes: set[tuple[int, ...]] = set()
    session_infos: list[dict[str, Any]] = []
    for audio_path in audio_paths:
        scenario = scenario_from_path(audio_path, smoke_dir)
        metadata = load_metadata(audio_path)
        info = audio_info(audio_path, {})
        audio = load_audio_mono(audio_path, sample_rate, {})
        session_id = str(metadata.get("session_id") or session_id_from_path(audio_path))
        session_key = f"{scenario}__{session_id}"
        ranges = iter_window_ranges(audio.size, sample_rate, window_sec, hop_sec)
        duration_sec = float(info.get("duration_sec") or (audio.size / sample_rate if sample_rate else 0.0))
        session_infos.append(
            {
                "session_key": session_key,
                "session_id": session_id,
                "scenario_hint": scenario,
                "session_type": session_type(scenario),
                "source_audio": str(audio_path),
                "metadata_json": str(audio_path.with_suffix(".json")) if audio_path.with_suffix(".json").exists() else "",
                "duration_sec": duration_sec,
                "sample_rate": int(info.get("sample_rate", sample_rate)),
                "channels": int(info.get("channels", 1)),
                "format": str(info.get("format", audio_path.suffix.lower())),
                "windows": int(len(ranges)),
                "freshness_note": freshness_note(audio_path, scenario),
            }
        )
        for window_index, (start_sample, end_sample) in enumerate(ranges):
            chunk = pad_window(audio[start_sample:end_sample], window_size)
            feature = log_mel_feature(chunk, audio_config)
            tensor = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
            shape = tuple(int(value) for value in tensor.shape)
            observed_shapes.add(shape)
            if list(shape) != expected_shape:
                raise ValueError(f"Observed input shape {list(shape)}, expected {expected_shape}")
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            cough_prob = float(probs[cough_index])
            row = {
                "smoke_dir": str(smoke_dir),
                "source_audio": str(audio_path),
                "metadata_json": str(audio_path.with_suffix(".json")) if audio_path.with_suffix(".json").exists() else "",
                "session_key": session_key,
                "session_id": session_id,
                "scenario_hint": scenario,
                "session_type": session_type(scenario),
                "freshness_note": freshness_note(audio_path, scenario),
                "audio_duration_sec": duration_sec,
                "audio_sample_rate": int(info.get("sample_rate", sample_rate)),
                "audio_channels": int(info.get("channels", 1)),
                "window_index": int(window_index),
                "window_start_sec": round(start_sample / sample_rate, 6),
                "window_end_sec": round(end_sample / sample_rate, 6),
                "fpfix_cough_prob": cough_prob,
            }
            for threshold in THRESHOLDS:
                row[f"trigger_ge_{threshold:.2f}".replace(".", "_")] = bool(cough_prob >= threshold)
            rows.append(row)
    scored = pd.DataFrame(rows)
    return scored, {
        "resolved_model_name": model_name,
        "class_order": class_names,
        "device_used": str(device),
        "expected_input_shape": expected_shape,
        "observed_input_shapes": [list(shape) for shape in sorted(observed_shapes)],
        "sessions": session_infos,
    }


def gate_flags(flags: list[bool], required: int, window_count: int) -> list[bool]:
    if window_count <= 1:
        return list(flags)
    output: list[bool] = []
    for index in range(len(flags)):
        if index + 1 < window_count:
            output.append(False)
            continue
        subset = flags[index - window_count + 1 : index + 1]
        output.append(sum(1 for value in subset if value) >= required)
    return output


def event_count(flags: list[bool]) -> int:
    count = 0
    previous = False
    for value in flags:
        if value and not previous:
            count += 1
        previous = value
    return count


def quantile(series: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.quantile(q)) if not numeric.empty else 0.0


def session_threshold_summary(scored: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for session_key, group in scored.groupby("session_key", sort=True):
        group = group.sort_values("window_index")
        duration_min = max(float(group["audio_duration_sec"].iloc[0]) / 60.0, 1e-9)
        session_rows: dict[str, Any] = {
            "scenario_hint": str(group["scenario_hint"].iloc[0]),
            "session_type": str(group["session_type"].iloc[0]),
            "freshness_note": str(group["freshness_note"].iloc[0]),
            "windows": int(len(group)),
            "duration_sec": float(group["audio_duration_sec"].iloc[0]),
            "max_cough_prob": float(pd.to_numeric(group["fpfix_cough_prob"]).max()),
            "p95_cough_prob": quantile(group["fpfix_cough_prob"], 0.95),
            "thresholds": {},
        }
        for threshold in THRESHOLDS:
            flags = pd.to_numeric(group["fpfix_cough_prob"]).ge(threshold).tolist()
            threshold_key = f"{threshold:.2f}"
            threshold_summary: dict[str, Any] = {}
            for gate_name, (required, gate_window_count) in GATES.items():
                gated = gate_flags(flags, required, gate_window_count)
                trigger_count = int(sum(gated))
                threshold_summary[gate_name] = {
                    "trigger_window_count": trigger_count,
                    "trigger_per_min": trigger_count / duration_min,
                    "event_count": event_count(gated),
                    "events_per_min": event_count(gated) / duration_min,
                }
            session_rows["thresholds"][threshold_key] = threshold_summary
        output[session_key] = session_rows
    return output


def scenario_summary(scored: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scenario, group in scored.groupby("scenario_hint", sort=True):
        output[str(scenario)] = {
            "sessions": int(group["session_key"].nunique()),
            "windows": int(len(group)),
            "session_type": str(group["session_type"].iloc[0]),
            "max_cough_prob": float(pd.to_numeric(group["fpfix_cough_prob"]).max()),
            "p95_cough_prob": quantile(group["fpfix_cough_prob"], 0.95),
            "freshness_notes": sorted(str(value) for value in group["freshness_note"].dropna().unique()),
        }
    return output


def top_suspicious_windows(scored: pd.DataFrame, per_session: int, overall: int) -> dict[str, Any]:
    non_cough = scored[scored["session_type"].eq("non_cough")].copy()
    columns = [
        "source_audio",
        "session_key",
        "scenario_hint",
        "freshness_note",
        "window_index",
        "window_start_sec",
        "window_end_sec",
        "fpfix_cough_prob",
    ]
    per_session_rows: dict[str, list[dict[str, Any]]] = {}
    for session_key, group in non_cough.groupby("session_key", sort=True):
        per_session_rows[str(session_key)] = (
            group.sort_values("fpfix_cough_prob", ascending=False)[columns].head(per_session).to_dict(orient="records")
        )
    return {
        "per_noncough_session": per_session_rows,
        "overall_noncough": non_cough.sort_values("fpfix_cough_prob", ascending=False)[columns]
        .head(overall)
        .to_dict(orient="records"),
    }


def load_annotations(smoke_dir: Path) -> pd.DataFrame | None:
    path = smoke_dir / "annotations.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"session_id", "label", "event_start_sec", "event_end_sec"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"annotations.csv missing columns: {missing}")
    return frame


def validate_scored(scored: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    if scored.empty:
        errors.append("scored CSV has no rows")
    probs = pd.to_numeric(scored.get("fpfix_cough_prob"), errors="coerce")
    if int(probs.isna().sum()):
        errors.append("fpfix_cough_prob has missing/non-numeric rows")
    if int((~probs.between(0.0, 1.0)).sum()):
        errors.append("fpfix_cough_prob has values outside [0, 1]")
    required = ["source_audio", "session_key", "scenario_hint", "window_start_sec", "window_end_sec"]
    for column in required:
        if column not in scored.columns:
            errors.append(f"missing required column {column}")
            continue
        empty = int(scored[column].astype(str).str.strip().eq("").sum())
        if empty:
            errors.append(f"{column} has {empty} empty rows")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "rows": int(len(scored)),
        "probability_range": {
            "min": float(probs.min()) if not probs.empty else None,
            "max": float(probs.max()) if not probs.empty else None,
        },
    }


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3.1 FP-fix Live Smoke PC Replay",
        "",
        f"- status: `{report['status']}`",
        f"- smoke_dir: `{report['smoke_dir']}`",
        f"- audio_files: `{report['audio_files']}`",
        f"- scored_windows: `{report['validation']['rows']}`",
        f"- model: `{report['resolved_model_name']}`",
        f"- annotations_available: `{report['annotations_available']}`",
        "",
        "## Scenario Summary",
        "",
    ]
    for scenario, item in report["scenario_summary"].items():
        lines.append(
            f"- `{scenario}` type `{item['session_type']}` windows `{item['windows']}` "
            f"max `{item['max_cough_prob']:.6f}` p95 `{item['p95_cough_prob']:.6f}` "
            f"freshness `{item['freshness_notes']}`"
        )
    lines.extend(["", "## Threshold/Gate Summary", ""])
    for session_key, item in report["session_threshold_summary"].items():
        lines.append(f"### {session_key}")
        lines.append(
            f"- scenario: `{item['scenario_hint']}`, type: `{item['session_type']}`, "
            f"duration_sec: `{item['duration_sec']:.3f}`, max: `{item['max_cough_prob']:.6f}`, "
            f"p95: `{item['p95_cough_prob']:.6f}`"
        )
        for threshold, threshold_item in item["thresholds"].items():
            single = threshold_item["single_window"]
            gate23 = threshold_item["2_of_3"]
            gate35 = threshold_item["3_of_5"]
            lines.append(
                f"- th `{threshold}` single `{single['trigger_window_count']}` "
                f"({single['trigger_per_min']:.3f}/min), 2of3 events `{gate23['event_count']}`, "
                f"3of5 events `{gate35['event_count']}`"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- This is a fresh/live smoke replay check, not a formal generalization evaluation.",
            "- Quiet-room data may include prior quiet background copies; see `freshness_note`.",
            "- No review clips were used.",
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
    config_path = resolve_path(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    smoke_dir = resolve_path(args.smoke_dir)
    out_path = resolve_path(args.out)
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {out_path}")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {report_path}")
    if not smoke_dir.exists():
        raise FileNotFoundError(f"Smoke directory not found: {smoke_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(config_path)
    frontend = frontend_summary(config)
    if frontend["status"] != "passed":
        raise ValueError(f"board_htk_no_norm_v1 validation failed: {frontend['checks']}")
    audio_paths = scan_audio_files(smoke_dir)
    scored, score_report = score_audio_files(audio_paths, smoke_dir, config, checkpoint_path)
    validation = validate_scored(scored)
    if validation["status"] != "passed":
        raise ValueError(f"scored validation failed: {validation['errors']}")

    annotations = load_annotations(smoke_dir)
    report = {
        "status": "passed",
        "smoke_dir": str(smoke_dir),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "out_csv": str(out_path),
        "report_json": str(report_path),
        "summary_md": str(summary_path),
        "audio_files": int(len(audio_paths)),
        "resolved_model_name": score_report["resolved_model_name"],
        "class_order": score_report["class_order"],
        "device_used": score_report["device_used"],
        "frontend": frontend,
        "expected_input_shape": score_report["expected_input_shape"],
        "observed_input_shapes": score_report["observed_input_shapes"],
        "sessions": score_report["sessions"],
        "validation": validation,
        "thresholds": THRESHOLDS,
        "gates": {name: {"required": value[0], "window_count": value[1]} for name, value in GATES.items()},
        "session_threshold_summary": session_threshold_summary(scored),
        "scenario_summary": scenario_summary(scored),
        "top_suspicious_windows": top_suspicious_windows(
            scored,
            per_session=args.top_windows_per_noncough_session,
            overall=args.top_windows_overall,
        ),
        "annotations_available": annotations is not None,
        "annotation_metrics_available": False,
        "annotation_note": (
            "annotations.csv not found; TP/FP/FN/TN are not reported. "
            "Use trigger/min and top suspicious windows for replay review."
            if annotations is None
            else "annotations.csv found but interval metric support is intentionally not used in this no-label smoke report."
        ),
        "generalization_claim": False,
        "generalization_note": "This smoke replay is not a formal generalization benchmark.",
        "not_executed": NOT_EXECUTED,
    }

    ensure_dir(out_path.parent)
    scored.to_csv(out_path, index=False)
    write_json(report, report_path)
    write_summary(report, summary_path)
    print(json.dumps({"status": "passed", "audio_files": len(audio_paths), "rows": len(scored)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
