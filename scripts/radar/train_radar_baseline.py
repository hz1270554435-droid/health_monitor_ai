from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.radar.train_tree import feature_importance_frame, train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train radar_assist v1 Random Forest baseline.")
    parser.add_argument("--config", default="configs/radar/radar_rf_v1.yaml")
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
    model, predictions, metrics = train_random_forest(features, config)

    model_path = resolve_path(paths["model_path"])
    predictions_path = resolve_path(paths["predictions_csv"])
    ensure_dir(model_path.parent)
    ensure_dir(predictions_path.parent)
    joblib.dump(model, model_path)
    predictions.to_csv(predictions_path, index=False)
    save_json(metrics, resolve_path(paths["metrics_json"]))

    importance = feature_importance_frame(model, metrics["feature_columns"])
    importance_path = resolve_path(paths["feature_importance_csv"])
    ensure_dir(importance_path.parent)
    importance.to_csv(importance_path, index=False)
    print(f"OK: trained radar Random Forest baseline; wrote predictions to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
