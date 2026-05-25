from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.evaluate import evaluate_checkpoint
from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained binary audio task model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    metrics = evaluate_checkpoint(
        config,
        run_dir=args.run_dir,
        model_path=args.model_path,
        manifest_path=args.manifest,
        split=args.split,
        threshold=args.threshold,
        root=ROOT,
    )
    print(f"OK: evaluated {metrics['rows']} {args.split} rows; f1={metrics['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
