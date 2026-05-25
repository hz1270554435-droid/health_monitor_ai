from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.radar.clean import clean_radar_frame, load_radar_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a parsed radar CSV for radar_assist v1.")
    parser.add_argument("--config", default="configs/radar/radar_features_v1.yaml")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    frame = load_radar_csv(resolve_path(paths["input_csv"]))
    cleaned, report = clean_radar_frame(frame, config["radar"])
    report["cleaned_columns"] = list(cleaned.columns)
    report["parsed_csv"] = str(resolve_path(paths["input_csv"]))
    report_path = resolve_path(paths["inspect_report"])
    ensure_dir(report_path.parent)
    save_json(report, report_path)
    print(f"OK: inspected {len(cleaned)} cleaned radar rows; wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
