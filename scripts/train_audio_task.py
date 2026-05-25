from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.train import run_training
from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a config-driven binary audio task model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = run_training(
        config,
        config_path=args.config,
        manifest_path=args.manifest,
        run_name=args.run_name,
        epochs=args.epochs,
        root=ROOT,
    )
    print(f"OK: checkpoint={paths['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
