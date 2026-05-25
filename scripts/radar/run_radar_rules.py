from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.radar.evaluate import evaluate_state_predictions
from src.radar.rules import apply_radar_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run radar_assist v1 rule baseline.")
    parser.add_argument("--config", default="configs/radar/radar_rules_v1.yaml")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    features_path = resolve_path(paths["features_csv"])
    if not features_path.exists():
        raise FileNotFoundError(f"Radar features CSV not found: {features_path}")
    features = pd.read_csv(features_path)
    output = apply_radar_rules(features, config)
    output_path = resolve_path(paths["output_csv"])
    ensure_dir(output_path.parent)
    output.to_csv(output_path, index=False)

    eval_cfg = config.get("evaluation", {})
    metrics = evaluate_state_predictions(
        output,
        target_column=str(eval_cfg.get("target_column", "radar_state_label")),
        prediction_column=str(eval_cfg.get("prediction_column", "radar_state")),
    )
    save_json(metrics, resolve_path(paths["metrics_json"]))
    print(f"OK: wrote {len(output)} radar rule rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
