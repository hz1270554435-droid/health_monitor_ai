from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir, save_json
from src.radar.clean import clean_radar_frame, load_radar_csv
from src.radar.windows import make_window_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create radar_assist v1 sliding windows from parsed radar CSV.")
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
    if windows.empty:
        raise ValueError("No radar windows were produced; check window_seconds, hop_seconds, and min_samples_per_window")
    windows_path = resolve_path(paths["windows_csv"])
    ensure_dir(windows_path.parent)
    windows.to_csv(windows_path, index=False)
    report_path = resolve_path(paths["results_dir"]) / "radar_windows_report.json"
    save_json(
        {
            "clean": clean_report,
            "window_count": int(len(windows)),
            "windows_csv": str(windows_path),
        },
        report_path,
    )
    print(f"OK: wrote {len(windows)} radar windows to {windows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
