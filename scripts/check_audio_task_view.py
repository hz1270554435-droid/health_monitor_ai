from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.audio.manifest import resolve_path
from src.audio.task_views import validate_task_view
from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an audio task-view CSV.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-view", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--require-two-classes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    path = resolve_path(args.task_view or config["paths"]["task_view_csv"], ROOT)
    if not path.exists():
        raise FileNotFoundError(f"Task view not found: {path}")
    frame = pd.read_csv(path)
    if args.max_rows:
        frame = frame.head(args.max_rows)
    validate_task_view(frame, config)
    label_counts = {str(k): int(v) for k, v in frame["task_label"].value_counts().sort_index().items()}
    split_counts = {str(k): int(v) for k, v in frame["split"].value_counts().sort_index().items()}
    errors: list[str] = []
    if args.require_two_classes and frame["label_id"].nunique() < 2:
        errors.append("Task view must contain both positive and negative classes")
    report = {
        "ok": not errors,
        "rows": int(len(frame)),
        "task": config["task"]["name"],
        "label_counts": label_counts,
        "split_counts": split_counts,
        "errors": errors,
    }
    out_path = resolve_path(args.out or f"results/{config['task']['name']}_task_view_report.json", ROOT)
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Report written to {out_path}")
        return 1
    print(f"OK: checked {len(frame)} rows; report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
