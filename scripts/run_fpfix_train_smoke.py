from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir


NOT_EXECUTED = [
    "formal training",
    "model export",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
    "board deployment",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-epoch FP-fix training smoke and write a compact report.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--run-name", default="audio_baseline_v3_1_fpfix_smoke")
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_train_smoke_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_train_smoke_summary.md")
    parser.add_argument("--skip-preprocess", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout.splitlines()[-40:],
        "stderr_tail": completed.stderr.splitlines()[-40:],
    }


def finite(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


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
        },
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v3.1 FP-fix One-Epoch Train Smoke",
        "",
        f"- status: `{report['status']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- model_dir: `{report['model_dir']}`",
        f"- epochs_run: `{report.get('epochs_run')}`",
        f"- checkpoint_generated: `{report['checkpoint_generated']}`",
        f"- sample_weight_active: `{report['sampler_status']['sample_weight_active']}`",
        f"- train WeightedRandomSampler: `{report['sampler_status']['train_weighted_sampler']}`",
        f"- val WeightedRandomSampler: `{report['sampler_status']['val_weighted_sampler']}`",
        f"- test WeightedRandomSampler: `{report['sampler_status']['test_weighted_sampler']}`",
        f"- train_loss: `{report.get('train_loss')}`",
        f"- val_loss: `{report.get('val_loss')}`",
        f"- loss_has_nan_or_inf: `{report['loss_has_nan_or_inf']}`",
        "",
        "## Split Rows",
        "",
        f"`{report['split_row_counts']}`",
        "",
        "## Val Metrics",
        "",
        f"`{report.get('val_metrics')}`",
        "",
        "## Test Metrics",
        "",
        f"`{report.get('test_metrics')}`",
        "",
        "## Frontend",
        "",
        f"`{report['board_htk_no_norm_v1']}`",
        "",
        "## Missing Audio / Feature Extraction",
        "",
        f"- missing_audio_or_feature_errors: `{report['missing_audio_or_feature_errors']}`",
        "",
        "## Commands",
        "",
    ]
    for item in report["commands_run"]:
        lines.append("- `" + " ".join(item["command"]) + f"` -> `{item['returncode']}`")
    lines.extend(["", "## Errors", ""])
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] if errors else ["- none"])
    lines.extend(["", "## Not Executed", ""])
    lines.extend([f"- {item}" for item in report["not_executed"]])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    labels_path = resolve_path(config["paths"]["labels_csv"])
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    manifest_path = processed_dir / "audio_manifest.csv"
    run_dir = resolve_path(config["paths"]["results_dir"]) / args.run_name
    model_dir = resolve_path(config["paths"]["models_dir"]) / args.run_name
    checkpoint_path = model_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.json"
    train_log_path = run_dir / "train_log.csv"
    split_manifest_path = run_dir / "split_manifest.csv"
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)

    commands: list[dict[str, Any]] = []
    errors: list[str] = []
    preprocess_result: dict[str, Any] | None = None

    if not args.skip_preprocess:
        preprocess_cmd = [
            sys.executable,
            "scripts/preprocess_audio.py",
            "--config",
            str(config_path),
            "--overwrite",
        ]
        preprocess_result = run_command(preprocess_cmd)
        commands.append(preprocess_result)
        if preprocess_result["returncode"] != 0:
            errors.append("preprocess_audio.py failed")

    train_result: dict[str, Any] | None = None
    if not errors:
        train_cmd = [
            sys.executable,
            "scripts/train_audio_baseline.py",
            "--config",
            str(config_path),
            "--run-name",
            args.run_name,
            "--epochs",
            "1",
        ]
        train_result = run_command(train_cmd)
        commands.append(train_result)
        if train_result["returncode"] != 0:
            errors.append("train_audio_baseline.py one-epoch smoke failed")

    train_log = pd.read_csv(train_log_path) if train_log_path.exists() else pd.DataFrame()
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    split_manifest = pd.read_csv(split_manifest_path) if split_manifest_path.exists() else pd.DataFrame()
    first_epoch = train_log.iloc[0].to_dict() if not train_log.empty else {}
    train_loss = first_epoch.get("train_loss")
    val_loss = first_epoch.get("val_loss")
    loss_has_nan_or_inf = not (finite(train_loss) and finite(val_loss))
    if loss_has_nan_or_inf:
        errors.append("train_loss or val_loss is NaN/Inf/non-numeric")

    sampler = metrics.get("sampler", {})
    train_sampler = sampler.get("train", {})
    val_sampler = sampler.get("val", {})
    test_sampler = sampler.get("test", {})
    sampler_status = {
        "sample_weight_active": bool(metrics.get("sample_weight_active", False)),
        "sampler_type": train_sampler.get("sampler_type"),
        "train_weighted_sampler": bool(train_sampler.get("uses_weighted_sampler", False)),
        "val_weighted_sampler": bool(val_sampler.get("uses_weighted_sampler", False)),
        "test_weighted_sampler": bool(test_sampler.get("uses_weighted_sampler", False)),
        "weight_column": metrics.get("weight_column"),
    }
    if not sampler_status["sample_weight_active"]:
        errors.append("sample_weight_active is false")
    if sampler_status["sampler_type"] != "WeightedRandomSampler":
        errors.append("train sampler is not WeightedRandomSampler")
    if not sampler_status["train_weighted_sampler"]:
        errors.append("train split did not use weighted sampler")
    if sampler_status["val_weighted_sampler"] or sampler_status["test_weighted_sampler"]:
        errors.append("val/test unexpectedly used weighted sampler")

    frontend = frontend_summary(config)
    if frontend["status"] != "passed":
        errors.append("board_htk_no_norm_v1 config check failed")

    checkpoint_generated = checkpoint_path.exists()
    if not checkpoint_generated:
        errors.append("checkpoint was not generated")

    split_counts = (
        {str(key): int(value) for key, value in split_manifest["split"].value_counts().sort_index().items()}
        if "split" in split_manifest.columns
        else {}
    )
    val_metrics = {
        "accuracy": first_epoch.get("val_accuracy"),
        "precision": first_epoch.get("val_precision"),
        "recall": first_epoch.get("val_recall"),
        "f1": first_epoch.get("val_f1"),
    }
    missing_audio_or_feature_errors = bool(
        preprocess_result is not None
        and preprocess_result["returncode"] != 0
        and (
            any("Audio file not found" in line for line in preprocess_result["stderr_tail"])
            or any("feature" in line.lower() for line in preprocess_result["stderr_tail"])
        )
    )

    report = {
        "status": "passed" if not errors else "failed",
        "command_run": commands[-1]["command"] if commands else [],
        "commands_run": commands,
        "resolved_config_path": str(config_path),
        "labels_path": str(labels_path),
        "processed_manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "epoch_count": 1,
        "epochs_run": metrics.get("epochs_run"),
        "split_row_counts": split_counts,
        "sampler_status": sampler_status,
        "model_name": metrics.get("model_name"),
        "class_names": metrics.get("class_names"),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_metrics": None,
        "train_metrics_available": False,
        "val_metrics": val_metrics,
        "test_metrics": metrics.get("test"),
        "checkpoint_generated": checkpoint_generated,
        "loss_has_nan_or_inf": bool(loss_has_nan_or_inf),
        "missing_audio_or_feature_errors": missing_audio_or_feature_errors,
        "board_htk_no_norm_v1": frontend,
        "one_epoch_smoke_run": bool(train_result is not None and train_result["returncode"] == 0),
        "formal_training_run": False,
        "model_export_run": False,
        "not_executed": NOT_EXECUTED,
        "errors": errors,
    }
    write_json(report, report_path)
    write_summary(report, summary_path)
    print(json.dumps({"status": report["status"], "report": str(report_path)}, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
