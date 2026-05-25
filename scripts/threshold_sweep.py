from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "health_monitor_ai_matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

from src.audio.dataset import FeatureDataset
from src.audio.models import build_model
from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep cough probability thresholds on a saved test split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("torch is required for threshold sweep") from exc
    return torch, DataLoader


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def resolve_feature_paths(manifest: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    output = manifest.copy()
    feature_root = manifest_path.parent
    output["feature_path"] = output["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    return output


def get_training_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("train") or config["training"]


def resolve_model_name(config: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict):
        training = get_training_section(checkpoint_config)
        if training.get("model"):
            return str(training["model"]).strip().lower()
    return str(get_training_section(config).get("model", "small_cnn")).strip().lower()


def threshold_metrics(y_true: list[int], cough_probs: list[float], threshold: float) -> dict[str, float | int]:
    y_pred = [1 if prob >= threshold else 0 for prob in cough_probs]
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    total = len(y_true)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def save_precision_recall_plot(frame: pd.DataFrame, path: Path, recommended_threshold: float | None) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        fallback_path = path.with_suffix(".txt")
        fallback_path.write_text(
            "matplotlib is not installed; precision/recall threshold plot was not generated.\n",
            encoding="utf-8",
        )
        return {"plot_generated": False, "plot_path": None, "fallback_path": str(fallback_path)}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame["threshold"], frame["precision"], marker="o", label="precision")
    ax.plot(frame["threshold"], frame["recall"], marker="o", label="recall")
    ax.plot(frame["threshold"], frame["f1"], marker="o", label="f1")
    if recommended_threshold is not None:
        ax.axvline(recommended_threshold, color="black", linestyle="--", linewidth=1, label="recommended")
    ax.set_xlabel("Cough probability threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"plot_generated": True, "plot_path": str(path), "fallback_path": None}


def select_thresholds(sweep: pd.DataFrame) -> dict[str, Any]:
    best_f1 = sweep.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    high_precision = sweep.sort_values(
        ["precision", "false_positive", "recall", "threshold"],
        ascending=[False, True, False, True],
    ).iloc[0]
    fp_candidates = sweep[sweep["threshold"].isin([0.90, 0.95])]
    if fp_candidates.empty:
        fp_candidates = sweep
    fp_control = fp_candidates.sort_values(
        ["false_positive", "recall", "precision", "threshold"],
        ascending=[True, False, False, False],
    ).iloc[0]
    return {
        "best_f1_threshold": float(best_f1["threshold"]),
        "high_precision_threshold": float(high_precision["threshold"]),
        "fp_control_threshold": float(fp_control["threshold"]),
    }


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def threshold_rows(sweep: pd.DataFrame, thresholds: list[float]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        match = sweep[sweep["threshold"].round(2).eq(round(threshold, 2))]
        if not match.empty:
            rows[f"{threshold:.2f}"] = {key: scalar(value) for key, value in match.iloc[0].to_dict().items()}
    return rows


def main() -> int:
    args = parse_args()
    torch, DataLoader = require_torch()
    config = load_config(args.config)

    run_dir = ensure_dir(resolve_path(args.run_dir))
    model_path = resolve_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    manifest_path = run_dir / "split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected saved split manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if "split" not in manifest.columns:
        raise ValueError(f"Saved split manifest has no split column: {manifest_path}")
    split_rows = manifest[manifest["split"] == args.split].reset_index(drop=True)
    if split_rows.empty:
        raise ValueError(f"No rows available for split={args.split} in {manifest_path}")
    split_rows = resolve_feature_paths(split_rows, manifest_path)

    training = get_training_section(config)
    device_name = training.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get(
        "class_names",
        [config["labels"]["negative"], config["labels"]["positive"]],
    )
    cough_label = config["labels"]["positive"]
    if cough_label not in class_names:
        raise ValueError(f"Checkpoint class_names does not include cough label '{cough_label}': {class_names}")
    cough_index = int(class_names.index(cough_label))

    resolved_model_name = resolve_model_name(config, checkpoint)
    model = build_model(resolved_model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    checkpoint_model_compatible = True
    model.eval()

    dataset = FeatureDataset(split_rows, root=manifest_path.parent).unwrap()
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
    )

    y_true: list[int] = []
    cough_probs: list[float] = []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            y_true.extend(int(value) for value in labels.tolist())
            cough_probs.extend(float(row[cough_index]) for row in probs)

    rows = [threshold_metrics(y_true, cough_probs, threshold / 100.0) for threshold in range(10, 100, 5)]
    sweep = pd.DataFrame(rows)
    candidates = sweep[sweep["recall"] >= 0.90].copy()
    recommended_threshold: float | None = None
    if not candidates.empty:
        candidates = candidates.sort_values(["f1", "threshold"], ascending=[False, True])
        recommended_threshold = float(candidates.iloc[0]["threshold"])
        sweep["recommended"] = sweep["threshold"].eq(recommended_threshold)
    else:
        sweep["recommended"] = False

    csv_path = run_dir / "threshold_sweep.csv"
    plot_path = run_dir / "precision_recall_threshold.png"
    sweep.to_csv(csv_path, index=False)
    plot_result = save_precision_recall_plot(sweep, plot_path, recommended_threshold)
    selected = select_thresholds(sweep)
    key_rows = threshold_rows(sweep, [0.50, 0.70, 0.80, 0.90, 0.95])
    summary_path = run_dir / "threshold_sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "rows": int(len(split_rows)),
                "model_path": str(model_path),
                "manifest": str(manifest_path),
                "resolved_model_name": resolved_model_name,
                "checkpoint_model_compatible": checkpoint_model_compatible,
                "threshold_sweep_csv": str(csv_path),
                "recommended_threshold_recall_ge_0_90": recommended_threshold,
                "best_f1_threshold": selected["best_f1_threshold"],
                "high_precision_threshold": selected["high_precision_threshold"],
                "fp_control_threshold": selected["fp_control_threshold"],
                "required_threshold_rows": key_rows,
                **plot_result,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"OK: wrote {csv_path}")
    if plot_result["plot_generated"]:
        print(f"OK: wrote {plot_path}")
    else:
        print(f"WARNING: wrote plot fallback {plot_result['fallback_path']}")
    print(f"OK: wrote {summary_path}")
    if recommended_threshold is None:
        print("WARNING: no threshold reached recall >= 0.90")
    else:
        best = sweep[sweep["recommended"]].iloc[0]
        print(
            "recommended_threshold="
            f"{recommended_threshold:.2f} "
            f"precision={best['precision']:.4f} "
            f"recall={best['recall']:.4f} "
            f"f1={best['f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
