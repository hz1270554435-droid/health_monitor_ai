from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run task-view check, preprocessing, training, and evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print(f"RUN: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    python = sys.executable
    config = load_config(args.config)
    run_name = args.run_name or f"{config['run']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config["paths"]["results_dir"]) / run_name
    model_path = Path(config["paths"]["models_dir"]) / run_name / "best_model.pt"
    processed_manifest = Path(config["paths"]["processed_dir"]) / "audio_manifest.csv"

    check_cmd = [python, "scripts/check_audio_task_view.py", "--config", args.config]
    preprocess_cmd = [python, "scripts/preprocess_audio.py", "--config", args.config]
    train_cmd = [python, "scripts/train_audio_task.py", "--config", args.config, "--run-name", run_name]
    eval_cmd = [
        python,
        "scripts/evaluate_audio_task.py",
        "--config",
        args.config,
        "--run-dir",
        str(run_dir),
        "--model-path",
        str(model_path),
        "--manifest",
        str(processed_manifest),
    ]

    if args.max_rows:
        check_cmd += ["--max-rows", str(args.max_rows)]
        preprocess_cmd += ["--limit", str(args.max_rows)]
    if args.overwrite:
        preprocess_cmd += ["--overwrite"]
    if args.epochs:
        train_cmd += ["--epochs", str(args.epochs)]

    print(f"run_dir={ROOT / run_dir}", flush=True)
    print(f"model_path={ROOT / model_path}", flush=True)
    run_step(check_cmd)
    run_step(preprocess_cmd)
    run_step(train_cmd)
    run_step(eval_cmd)
    print(f"OK: pipeline completed for run_name={run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
