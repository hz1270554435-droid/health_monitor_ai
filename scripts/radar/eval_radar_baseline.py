from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import save_json
from src.radar.evaluate import evaluate_state_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate radar_assist rule or small-model predictions.")
    parser.add_argument("--config", default="configs/radar/radar_rules_v1.yaml")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    predictions_key = "predictions_csv" if "predictions_csv" in paths else "output_csv"
    predictions_path = resolve_path(paths[predictions_key])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Radar prediction CSV not found: {predictions_path}")
    frame = pd.read_csv(predictions_path)
    eval_cfg = config.get("evaluation", {})
    prediction_column = str(
        eval_cfg.get("prediction_column", "pred_radar_state" if predictions_key == "predictions_csv" else "radar_state")
    )
    metrics = evaluate_state_predictions(
        frame,
        target_column=str(eval_cfg.get("target_column", "radar_state_label")),
        prediction_column=prediction_column,
    )
    metrics_path = resolve_path(paths["metrics_json"])
    save_json(metrics, metrics_path)
    print(f"OK: evaluated {len(frame)} radar rows; wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
