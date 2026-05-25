from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.io import ensure_dir
from src.radar.export_rules import generate_rules_header, generate_rules_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export radar_assist v1 rules as C header/source files.")
    parser.add_argument("--config", default="configs/radar/radar_rules_v1.yaml")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    header_path = resolve_path(paths["rules_header"])
    source_path = resolve_path(paths["rules_source"])
    ensure_dir(header_path.parent)
    ensure_dir(source_path.parent)
    header_path.write_text(generate_rules_header(), encoding="utf-8")
    source_path.write_text(generate_rules_source(config, header_path), encoding="utf-8")
    print(f"OK: exported radar rules to {header_path} and {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
