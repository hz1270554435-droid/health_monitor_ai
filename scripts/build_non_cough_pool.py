from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.audio.manifest import build_audio_manifest, resolve_path, write_manifest
from src.audio.task_views import build_non_cough_pool
from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared non_cough negative pool index.")
    parser.add_argument("--config", default="configs/non_cough_pool.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.manifest:
        manifest_path = resolve_path(args.manifest, ROOT)
    else:
        manifest_path = resolve_path(config["paths"]["audio_manifest_csv"], ROOT)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_audio_manifest(config, root=ROOT)
        write_manifest(manifest, manifest_path, root=ROOT)
    pool = build_non_cough_pool(manifest, config)
    out_path = args.out or config["paths"]["non_cough_pool_csv"]
    target = write_manifest(pool, out_path, root=ROOT)
    print(f"OK: wrote {len(pool)} shared non_cough pool rows to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
