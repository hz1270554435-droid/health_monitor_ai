from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.manifest import build_audio_manifest, validate_manifest, write_manifest
from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical audio clip/interval manifest.")
    parser.add_argument("--config", default="configs/non_cough_pool.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--require-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest = build_audio_manifest(config, root=ROOT)
    validate_manifest(manifest, require_files=args.require_files, root=ROOT)
    out_path = args.out or config["paths"]["audio_manifest_csv"]
    target = write_manifest(manifest, out_path, root=ROOT)
    print(f"OK: wrote {len(manifest)} audio manifest rows to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
