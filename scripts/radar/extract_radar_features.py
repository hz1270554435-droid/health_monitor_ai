from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.radar.clean import clean_radar_frame, load_radar_csv
from src.radar.features import extract_window_features
from src.radar.windows import make_window_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract radar_assist v1 sliding-window features.")
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
    cleaned, clean_report = clean_radar_frame(frame, config["radar"])
    windows = make_window_table(cleaned, config["radar"])
    features = extract_window_features(cleaned, windows, config)

    windows_path = resolve_path(paths["windows_csv"])
    features_path = resolve_path(paths["features_csv"])
    ensure_dir(windows_path.parent)
    ensure_dir(features_path.parent)
    windows.to_csv(windows_path, index=False)
    features.to_csv(features_path, index=False)
    save_json(
        {
            "clean": clean_report,
            "window_count": int(len(windows)),
            "feature_rows": int(len(features)),
            "features_csv": str(features_path),
        },
        resolve_path(paths["results_dir"]) / "radar_features_report.json",
    )
    print(f"OK: wrote {len(features)} radar feature rows to {features_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
