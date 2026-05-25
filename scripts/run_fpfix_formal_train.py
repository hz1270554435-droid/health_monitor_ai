from __future__ import annotations

import argparse
import json
import math
import shutil
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
    "model export",
    "board deployment",
    "raw-data modification",
    "final label merge",
    "OPERA/HeAR/YAMNet distillation",
]

REFERENCE_NOTE = (
    "v3 baseline comparable metric is read for reference only; final FP-fix effectiveness "
    "requires regression eval on board_live_data_1 and fresh board-live smoke."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run formal v3.1 FP-fix training and summarize validation.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_1_fpfix.yaml")
    parser.add_argument("--run-name", default="audio_baseline_v3_1_fpfix")
    parser.add_argument("--report", default="review/board_live_data_1/fpfix_train_report.json")
    parser.add_argument("--summary-md", default="review/board_live_data_1/fpfix_train_summary.md")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-threshold-sweep", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "stdout_tail": completed.stdout.splitlines()[-80:],
        "stderr_tail": completed.stderr.splitlines()[-80:],
    }


def get_training_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train") or config["training"]


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


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    training = get_training_section(config)
    sampling = config.get("sampling", {})
    frontend = frontend_summary(config)
    if frontend["status"] != "passed":
        errors.append("board_htk_no_norm_v1 config check failed")
    if str(training.get("model")).strip().lower() != "ds_cnn":
        errors.append(f"expected training.model=ds_cnn, got {training.get('model')}")
    if str(sampling.get("mode")).strip().lower() != "weighted_random":
        errors.append(f"expected sampling.mode=weighted_random, got {sampling.get('mode')}")
    if not bool(sampling.get("use_sample_weight", False)):
        errors.append("expected sampling.use_sample_weight=true")
    if str(sampling.get("weight_column", "")).strip() != "sample_weight":
        errors.append(f"expected sampling.weight_column=sample_weight, got {sampling.get('weight_column')}")
    return errors


def safe_remove_output_dir(path: Path, expected_parent: Path, run_name: str) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_parent = expected_parent.resolve()
    if resolved_path.parent != resolved_parent or resolved_path.name != run_name:
        raise ValueError(f"Refusing to remove unexpected output dir: {path}")
    shutil.rmtree(resolved_path)


def protect_output_dirs(run_dir: Path, model_dir: Path, run_name: str, overwrite: bool) -> None:
    existing = [path for path in (run_dir, model_dir) if path.exists()]
    if not existing:
        return
    if not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Formal output dir exists; pass --overwrite to replace: {joined}")
    safe_remove_output_dir(run_dir, run_dir.parent, run_name)
    safe_remove_output_dir(model_dir, model_dir.parent, run_name)


def load_train_log(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def loss_has_nan_or_inf(train_log: pd.DataFrame) -> bool:
    if train_log.empty:
        return True
    for column in ("train_loss", "val_loss"):
        if column not in train_log.columns:
            return True
        if not all(finite(value) for value in train_log[column].tolist()):
            return True
    return False


def confusion_matrix_path(run_dir: Path) -> str | None:
    for name in ("confusion_matrix.png", "confusion_matrix.txt"):
        candidate = run_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def threshold_rows_for_report(sweep_path: Path) -> dict[str, dict[str, Any]]:
    if not sweep_path.exists():
        return {}
    sweep = pd.read_csv(sweep_path)
    rows: dict[str, dict[str, Any]] = {}
    for threshold in (0.50, 0.70, 0.80, 0.90, 0.95):
        match = sweep[sweep["threshold"].round(2).eq(round(threshold, 2))]
        if not match.empty:
            rows[f"{threshold:.2f}"] = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in match.iloc[0].to_dict().items()
            }
    return rows


def load_v3_baseline_reference() -> dict[str, Any]:
    metrics_path = ROOT / "results" / "audio" / "audio_baseline_v3_board_htk_hardneg" / "metrics.json"
    if not metrics_path.exists():
        return {"available": False, "note": REFERENCE_NOTE}
    metrics = read_json(metrics_path)
    return {
        "available": True,
        "metrics_path": str(metrics_path),
        "best_epoch": metrics.get("best_epoch"),
        "best_metric": metrics.get("best_metric"),
        "test": metrics.get("test"),
        "note": REFERENCE_NOTE,
    }


def staged_generated_artifacts(status_text: str) -> list[str]:
    prefixes = (
        "data/processed/audio_features_v3_1_fpfix/",
        "models/audio/audio_baseline_v3_1_fpfix/",
        "results/audio/audio_baseline_v3_1_fpfix/",
    )
    staged: list[str] = []
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        index_status = line[0]
        path = line[3:].replace("\\", "/")
        if index_status != " " and any(path.startswith(prefix) for prefix in prefixes):
            staged.append(line)
    return staged


def write_summary(report: dict[str, Any], path: Path) -> None:
    threshold = report.get("threshold_sweep", {})
    lines = [
        "# v3.1 FP-fix Formal Training",
        "",
        f"- status: `{report['status']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- model_dir: `{report['model_dir']}`",
        f"- checkpoint: `{report['best_checkpoint_path']}`",
        f"- epochs_configured: `{report['epochs_configured']}`",
        f"- epochs_run: `{report.get('epochs_run')}`",
        f"- best_epoch: `{report.get('best_epoch')}`",
        f"- sample_weight_active: `{report['sampler_status']['sample_weight_active']}`",
        f"- train WeightedRandomSampler: `{report['sampler_status']['train_weighted_sampler']}`",
        f"- val WeightedRandomSampler: `{report['sampler_status']['val_weighted_sampler']}`",
        f"- test WeightedRandomSampler: `{report['sampler_status']['test_weighted_sampler']}`",
        f"- resolved_model_name_train: `{report['resolved_model_name_train']}`",
        f"- resolved_model_name_eval: `{report['resolved_model_name_eval']}`",
        f"- resolved_model_name_threshold_sweep: `{report['resolved_model_name_threshold_sweep']}`",
        f"- checkpoint_model_compatible: `{report['checkpoint_model_compatible']}`",
        f"- loss_has_nan_or_inf: `{report['loss_has_nan_or_inf']}`",
        "",
        "## Metrics",
        "",
        f"- final_train_loss: `{report.get('final_train_loss')}`",
        f"- final_val_loss: `{report.get('final_val_loss')}`",
        f"- best_metric: `{report.get('best_metric')}`",
        f"- test_metrics: `{report.get('test_metrics')}`",
        "",
        "## Threshold Sweep",
        "",
        f"- threshold_sweep_csv: `{threshold.get('threshold_sweep_csv')}`",
        f"- best_f1_threshold: `{threshold.get('best_f1_threshold')}`",
        f"- high_precision_threshold: `{threshold.get('high_precision_threshold')}`",
        f"- fp_control_threshold: `{threshold.get('fp_control_threshold')}`",
        f"- required_threshold_rows: `{threshold.get('required_threshold_rows')}`",
        "",
        "## v3 Baseline Reference",
        "",
        f"- {report['v3_baseline_reference']['note']}",
        f"- metrics: `{report['v3_baseline_reference']}`",
        "",
        "## Generated Artifacts Not For Git",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["generated_artifacts_not_for_git"]])
    lines.extend(["", "## Not Executed", ""])
    lines.extend([f"- {item}" for item in report["not_executed"]])
    lines.extend(["", "## Errors", ""])
    errors = report.get("errors", [])
    lines.extend([f"- {error}" for error in errors] if errors else ["- none"])
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    training = get_training_section(config)
    labels_path = resolve_path(config["paths"]["labels_csv"])
    processed_dir = resolve_path(config["paths"]["processed_dir"])
    manifest_path = processed_dir / "audio_manifest.csv"
    run_dir = resolve_path(config["paths"]["results_dir"]) / args.run_name
    model_dir = resolve_path(config["paths"]["models_dir"]) / args.run_name
    checkpoint_path = model_dir / "best_model.pt"
    report_path = resolve_path(args.report)
    summary_path = resolve_path(args.summary_md)
    eval_dir = run_dir / "eval_test_default"
    metrics_path = run_dir / "metrics.json"
    train_log_path = run_dir / "train_log.csv"
    split_manifest_path = run_dir / "split_manifest.csv"
    threshold_summary_path = run_dir / "threshold_sweep_summary.json"
    threshold_csv_path = run_dir / "threshold_sweep.csv"

    commands: list[dict[str, Any]] = []
    errors = validate_config(config)
    if not labels_path.exists():
        errors.append(f"labels file not found: {labels_path}")
    try:
        protect_output_dirs(run_dir, model_dir, args.run_name, args.overwrite)
    except (FileExistsError, ValueError) as exc:
        errors.append(str(exc))

    if not errors and not args.skip_preprocess:
        preprocess_result = run_command(
            [sys.executable, "scripts/preprocess_audio.py", "--config", str(config_path), "--overwrite"]
        )
        commands.append(preprocess_result)
        if preprocess_result["returncode"] != 0:
            errors.append("preprocess_audio.py failed")

    if not errors:
        train_result = run_command(
            [sys.executable, "scripts/train_audio_baseline.py", "--config", str(config_path), "--run-name", args.run_name]
        )
        commands.append(train_result)
        if train_result["returncode"] != 0:
            errors.append("train_audio_baseline.py formal training failed")

    if not errors:
        eval_result = run_command(
            [
                sys.executable,
                "scripts/evaluate_model.py",
                "--config",
                str(config_path),
                "--run-dir",
                str(eval_dir),
                "--model-path",
                str(checkpoint_path),
                "--manifest",
                str(split_manifest_path),
                "--split",
                "test",
            ]
        )
        commands.append(eval_result)
        if eval_result["returncode"] != 0:
            errors.append("evaluate_model.py failed")

    if not errors and not args.skip_threshold_sweep:
        sweep_result = run_command(
            [
                sys.executable,
                "scripts/threshold_sweep.py",
                "--config",
                str(config_path),
                "--run-dir",
                str(run_dir),
                "--model-path",
                str(checkpoint_path),
                "--split",
                "test",
            ]
        )
        commands.append(sweep_result)
        if sweep_result["returncode"] != 0:
            errors.append("threshold_sweep.py failed")

    train_log = load_train_log(train_log_path)
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    eval_metrics_path = eval_dir / "metrics.json"
    eval_metrics = read_json(eval_metrics_path) if eval_metrics_path.exists() else {}
    threshold_summary = read_json(threshold_summary_path) if threshold_summary_path.exists() else {}
    split_manifest = pd.read_csv(split_manifest_path) if split_manifest_path.exists() else pd.DataFrame()
    final_epoch = train_log.iloc[-1].to_dict() if not train_log.empty else {}

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
    if metrics and not sampler_status["sample_weight_active"]:
        errors.append("sample_weight_active is false")
    if metrics and sampler_status["sampler_type"] != "WeightedRandomSampler":
        errors.append("train sampler is not WeightedRandomSampler")
    if metrics and not sampler_status["train_weighted_sampler"]:
        errors.append("train split did not use weighted sampler")
    if metrics and (sampler_status["val_weighted_sampler"] or sampler_status["test_weighted_sampler"]):
        errors.append("val/test unexpectedly used weighted sampler")

    frontend = frontend_summary(config)
    if frontend["status"] != "passed" and "board_htk_no_norm_v1 config check failed" not in errors:
        errors.append("board_htk_no_norm_v1 config check failed")

    has_bad_loss = loss_has_nan_or_inf(train_log)
    if not train_log.empty and has_bad_loss:
        errors.append("train_loss or val_loss is NaN/Inf/non-numeric")
    if not checkpoint_path.exists() and metrics:
        errors.append("best_model.pt was not generated")

    threshold = {
        "threshold_sweep_csv": str(threshold_csv_path) if threshold_csv_path.exists() else None,
        "best_f1_threshold": threshold_summary.get("best_f1_threshold"),
        "high_precision_threshold": threshold_summary.get("high_precision_threshold"),
        "fp_control_threshold": threshold_summary.get("fp_control_threshold"),
        "required_threshold_rows": threshold_summary.get("required_threshold_rows") or threshold_rows_for_report(threshold_csv_path),
        "plot_generated": threshold_summary.get("plot_generated"),
        "plot_path": threshold_summary.get("plot_path"),
        "plot_fallback_path": threshold_summary.get("fallback_path"),
    }
    missing_thresholds = sorted(set(["0.50", "0.70", "0.80", "0.90", "0.95"]) - set(threshold["required_threshold_rows"]))
    if threshold_csv_path.exists() and missing_thresholds:
        errors.append(f"threshold sweep missing required threshold rows: {missing_thresholds}")

    status_result = run_command(["git", "status", "--short"])
    staged_artifacts = staged_generated_artifacts("\n".join(status_result["stdout_tail"]))
    if staged_artifacts:
        errors.append("generated model/results/processed artifacts are staged")

    split_counts = (
        {str(key): int(value) for key, value in split_manifest["split"].value_counts().sort_index().items()}
        if "split" in split_manifest.columns
        else {}
    )
    model_names = {
        "train": metrics.get("model_name"),
        "eval": eval_metrics.get("resolved_model_name"),
        "threshold_sweep": threshold_summary.get("resolved_model_name"),
    }
    checkpoint_compatible = bool(
        checkpoint_path.exists()
        and eval_metrics.get("checkpoint_model_compatible") is True
        and threshold_summary.get("checkpoint_model_compatible") is True
    )
    if metrics and any(value != "ds_cnn" for value in model_names.values()):
        errors.append(f"resolved model names are not all ds_cnn: {model_names}")
    if eval_metrics and threshold_summary and not checkpoint_compatible:
        errors.append("checkpoint_model_compatible is false")

    report = {
        "status": "passed" if not errors else "failed",
        "commands_run": commands,
        "resolved_config_path": str(config_path),
        "labels_path": str(labels_path),
        "processed_manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "model_dir": str(model_dir),
        "epochs_configured": int(training["epochs"]),
        "epochs_run": metrics.get("epochs_run"),
        "best_epoch": metrics.get("best_epoch"),
        "best_metric": metrics.get("best_metric"),
        "best_checkpoint_path": str(checkpoint_path),
        "checkpoint_generated": checkpoint_path.exists(),
        "split_row_counts": split_counts,
        "sampler_status": sampler_status,
        "resolved_model_name_train": model_names["train"],
        "resolved_model_name_eval": model_names["eval"],
        "resolved_model_name_threshold_sweep": model_names["threshold_sweep"],
        "checkpoint_model_compatible": checkpoint_compatible,
        "final_train_loss": final_epoch.get("train_loss"),
        "final_val_loss": final_epoch.get("val_loss"),
        "final_val_metrics": {
            "accuracy": final_epoch.get("val_accuracy"),
            "precision": final_epoch.get("val_precision"),
            "recall": final_epoch.get("val_recall"),
            "f1": final_epoch.get("val_f1"),
        },
        "test_metrics": metrics.get("test"),
        "eval_test_metrics": {
            key: eval_metrics.get(key) for key in ("accuracy", "precision", "recall", "f1")
        },
        "threshold_sweep": threshold,
        "confusion_matrix_path": confusion_matrix_path(run_dir),
        "eval_confusion_matrix_path": confusion_matrix_path(eval_dir),
        "loss_has_nan_or_inf": has_bad_loss,
        "missing_audio_or_feature_errors": any(
            line for command in commands for line in command["stderr_tail"] if "Audio file not found" in line
        ),
        "board_htk_no_norm_v1": frontend,
        "v3_baseline_reference": load_v3_baseline_reference(),
        "generalization_note": "p01 addon is in train only; this run cannot prove generalization.",
        "generated_artifacts_not_for_git": [
            "ml/data/processed/audio_features_v3_1_fpfix/",
            "ml/models/audio/audio_baseline_v3_1_fpfix/",
            "ml/results/audio/audio_baseline_v3_1_fpfix/",
        ],
        "generated_artifacts_staged": staged_artifacts,
        "git_status_short_tail": status_result["stdout_tail"][-80:],
        "formal_training_run": bool(metrics.get("epochs_run")),
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
