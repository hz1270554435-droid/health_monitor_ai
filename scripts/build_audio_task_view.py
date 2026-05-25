from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.audio.manifest import resolve_path
from src.audio.task_views import build_task_view, validate_task_view
from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a binary audio task view from the canonical manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--non-cough-pool", default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    manifest_path = resolve_path(args.manifest or paths["audio_manifest_csv"], ROOT)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Audio manifest not found: {manifest_path}. "
            "Run scripts/build_audio_manifest.py first."
        )
    manifest = pd.read_csv(manifest_path)

    pool_path = resolve_path(args.non_cough_pool or paths.get("non_cough_pool_csv", ""), ROOT)
    non_cough_pool = pd.read_csv(pool_path) if str(pool_path) and pool_path.exists() else None
    view = build_task_view(manifest, config, non_cough_pool=non_cough_pool)
    validate_task_view(view, config)
    out_path = resolve_path(args.out or paths["task_view_csv"], ROOT)
    ensure_dir(out_path.parent)
    view.to_csv(out_path, index=False)
    counts = view["task_label"].value_counts().to_dict()
    print(f"OK: wrote {len(view)} task rows to {out_path}; labels={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
