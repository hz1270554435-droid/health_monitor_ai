from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.evaluate import run_threshold_sweep
from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep positive-label probability thresholds for an audio task.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sweep = run_threshold_sweep(
        config,
        run_dir=args.run_dir,
        model_path=args.model_path,
        split=args.split,
        root=ROOT,
    )
    recommended = sweep[sweep["recommended"]]
    print(f"OK: wrote threshold sweep to {Path(args.run_dir) / 'threshold_sweep.csv'}")
    if recommended.empty:
        print("WARNING: no threshold satisfied the configured recall constraint")
    else:
        row = recommended.iloc[0]
        print(
            f"recommended_threshold={row['threshold']:.2f} "
            f"precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
