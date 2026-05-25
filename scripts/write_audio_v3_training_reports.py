from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write v3 audio baseline experiment and follow-up reports.")
    parser.add_argument("--config", default="configs/audio_baseline_v3_board_htk_hardneg.yaml")
    parser.add_argument("--run-dir", default="results/audio/audio_baseline_v3_board_htk_hardneg")
    parser.add_argument("--dataset-check", default="results/check_audio_labels_v3/report.json")
    parser.add_argument("--pretrain-check", default="results/pretrain_check_audio_labels_v3/report.json")
    parser.add_argument("--pretrain-preprocess", default="results/pretrain_preprocess_smoke_v3/report.json")
    parser.add_argument("--pretrain-one-epoch", default="results/pretrain_one_epoch_smoke_v3/metrics.json")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def read_config(path: str | Path) -> dict[str, Any]:
    with resolve(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_None_\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def recommended_threshold(sweep: pd.DataFrame) -> dict[str, Any] | None:
    if "recommended" not in sweep.columns:
        return None
    recommended = sweep[sweep["recommended"].astype(bool)]
    if recommended.empty:
        return None
    return recommended.iloc[0].to_dict()


def best_f1_threshold(sweep: pd.DataFrame) -> dict[str, Any] | None:
    if sweep.empty:
        return None
    return sweep.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0].to_dict()


def write_experiment_report(
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    dataset: dict[str, Any],
    pretrain_check: dict[str, Any],
    pretrain_preprocess: dict[str, Any],
    pretrain_one_epoch: dict[str, Any],
    sweep: pd.DataFrame,
    classification_report: str,
) -> None:
    rec = recommended_threshold(sweep)
    best = best_f1_threshold(sweep)
    audio = config["audio"]
    training = config["training"]
    sampling = config.get("sampling", {})
    lines = [
        "# audio_baseline_v3_board_htk_hardneg Experiment Report",
        "",
        "## Scope",
        "",
        "This run trains a DS-CNN cough/non_cough candidate model using the finalized v3 labels. "
        "It is not an OPERA/distillation run and should not be treated as final generalization evidence.",
        "",
        "## Inputs",
        "",
        f"- labels: `{config['paths']['labels_csv']}`",
        f"- processed_dir: `{config['paths']['processed_dir']}`",
        f"- checkpoint: `{metrics.get('checkpoint')}`",
        f"- frontend: `{audio.get('frontend', 'board_htk_no_norm_v1')}`",
        f"- sample_rate: `{audio['sample_rate']}`",
        f"- window/hop: `{audio['window_seconds']}s / {audio['hop_seconds']}s`",
        f"- Log-Mel: `{audio['n_mels']} bins`, HTK=`{audio.get('htk')}`, normalize=`{audio.get('normalize')}`",
        "",
        "## Training Setup",
        "",
        f"- model: `{training.get('model')}`",
        f"- class_weight: `{training.get('class_weight')}`",
        f"- sampler mode: `{sampling.get('mode')}`",
        f"- hard_negative_oversample: `{sampling.get('hard_negative_oversample')}`",
        f"- hard_negative_multiplier: `{sampling.get('hard_negative_multiplier')}`",
        f"- sample_weight used by sampler: `{sampling.get('use_sample_weight')}`",
        f"- epochs_run: `{metrics.get('epochs_run')}`",
        f"- best_epoch: `{metrics.get('best_epoch')}`",
        f"- early_stop_triggered: `{metrics.get('early_stop_triggered')}`",
        "",
        "## Pre-Training Checks",
        "",
        f"- dataset check: `{pretrain_check.get('status')}`",
        f"- preprocess smoke: `{pretrain_preprocess.get('status')}`",
        f"- one-epoch smoke: `{pretrain_one_epoch.get('status')}`",
        f"- finalized dataset check: `{dataset.get('status')}`",
        "",
        "## Data Summary",
        "",
        f"- label_counts: `{dataset.get('label_counts')}`",
        f"- split_counts labels: `{dataset.get('split_counts')}`",
        f"- hard_negative_count: `{dataset.get('hard_negative_count')}`",
        f"- hard_negative_source_counts: `{dataset.get('hard_negative_source_counts')}`",
        f"- training manifest split_counts: `{metrics.get('split_counts')}`",
        "",
        "## Test Metrics",
        "",
        f"- argmax/0.50 accuracy: `{metrics.get('test', {}).get('accuracy')}`",
        f"- argmax/0.50 precision: `{metrics.get('test', {}).get('precision')}`",
        f"- argmax/0.50 recall: `{metrics.get('test', {}).get('recall')}`",
        f"- argmax/0.50 f1: `{metrics.get('test', {}).get('f1')}`",
        "",
        "## Classification Report",
        "",
        "```text",
        classification_report.strip(),
        "```",
        "",
        "## Threshold Sweep",
        "",
    ]
    if rec is None:
        lines.append("- recommended threshold: none satisfied configured recall constraint")
    else:
        lines.append(
            "- recommended threshold under recall constraint: "
            f"`{rec['threshold']:.2f}` precision=`{rec['precision']:.4f}` "
            f"recall=`{rec['recall']:.4f}` f1=`{rec['f1']:.4f}`"
        )
    if best is not None:
        lines.append(
            "- best observed F1 threshold on current tiny test: "
            f"`{best['threshold']:.2f}` precision=`{best['precision']:.4f}` "
            f"recall=`{best['recall']:.4f}` f1=`{best['f1']:.4f}`"
        )
    lines.extend(
        [
            "",
            md_table(
                sweep,
                [
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "false_positive",
                    "false_negative",
                    "recommended",
                ],
            ),
            "## Interpretation",
            "",
            "The current run is useful for selecting a final candidate and board-side test threshold. "
            "Because val/test are small and do not yet include independent hard-negative coverage, "
            "do not claim final generalization from these numbers.",
            "",
        ]
    )
    (run_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_board_test_plan(run_dir: Path, config: dict[str, Any], metrics: dict[str, Any], sweep: pd.DataFrame) -> None:
    rec = recommended_threshold(sweep)
    threshold_text = f"{rec['threshold']:.2f}" if rec is not None else "0.75"
    lines = [
        "# Board Test Plan",
        "",
        "## Goal",
        "",
        "Verify that `audio_baseline_v3_board_htk_hardneg` behaves consistently when deployed with the board-compatible audio frontend.",
        "",
        "## Frozen Assumptions",
        "",
        f"- frontend: `{config['audio'].get('frontend', 'board_htk_no_norm_v1')}`",
        f"- sample_rate: `{config['audio']['sample_rate']}`",
        f"- window/hop: `{config['audio']['window_seconds']}s / {config['audio']['hop_seconds']}s`",
        f"- n_mels: `{config['audio']['n_mels']}`",
        f"- normalize: `{config['audio'].get('normalize')}`",
        f"- model: `{config['training'].get('model')}`",
        f"- candidate threshold to test first: `{threshold_text}`",
        f"- checkpoint: `{metrics.get('checkpoint')}`",
        "",
        "## Required Offline Checks Before Flash",
        "",
        "1. Export the checkpoint to the deployment format used by the current firmware path.",
        "2. Generate fixed audio test vectors from cough, clean non_cough, model_false_alarm hard negatives, and human_cough_like hard negatives.",
        "3. Compare Python frontend features against board/frontend output on the same 1 s windows.",
        "4. Confirm probability differences stay within a documented tolerance before judging board behavior.",
        "",
        "## Minimum Board Clip Set",
        "",
        "- 20 clear cough clips from existing val/test or newly held-out cough data.",
        "- 20 clean non_cough clips: speech, silence, music, ambient appliance sound.",
        "- 20 model_false_alarm hard negatives: sudden loud changes, music hits, screams/shouts, impulse noise.",
        "- 20 human_cough_like hard negatives: throat clearing, sneeze-like, breath/noise bursts if available.",
        "- 10 low-SNR/no-signal clips that should be excluded or suppressed by upstream quality logic.",
        "",
        "## Board Metrics To Record",
        "",
        "- Per-window cough probability.",
        "- Trigger decision at thresholds 0.50, 0.75, 0.90, and candidate threshold.",
        "- False positive events after event merging.",
        "- Latency and memory headroom.",
        "- Any frontend mismatch against Python features.",
        "",
        "## Pass Gate For Firmware Candidate",
        "",
        "This model can become a firmware candidate only after the board run reproduces Python behavior on fixed vectors and the supplemental independent val/test set confirms acceptable recall/false alarm tradeoff.",
        "",
    ]
    (run_dir / "board_test_plan.md").write_text("\n".join(lines), encoding="utf-8")


def write_eval_gap_report(run_dir: Path, dataset: dict[str, Any], metrics: dict[str, Any], sweep: pd.DataFrame) -> None:
    rec = recommended_threshold(sweep)
    rec_line = "none"
    if rec is not None:
        rec_line = (
            f"{rec['threshold']:.2f} with precision={rec['precision']:.4f}, "
            f"recall={rec['recall']:.4f}, f1={rec['f1']:.4f}"
        )
    lines = [
        "# Evaluation Data Gap Report",
        "",
        "## Current Limitation",
        "",
        f"- Label split counts are `{dataset.get('split_counts')}`.",
        f"- Feature-window split counts are `{metrics.get('split_counts')}`.",
        "- The validation and test sets are too small for a final generalization claim.",
        "- Most new YAMNet-mined hard negatives were intentionally added to train, so val/test do not yet reflect the new deployment risk distribution.",
        "",
        "## How To Interpret This Run",
        "",
        f"- Recommended threshold from the current tiny test sweep: `{rec_line}`.",
        "- Treat this as final candidate selection evidence, not as final product validation.",
        "- The result is most useful for deciding whether the v3 hard-negative training recipe is worth testing on independent held-out data.",
        "",
        "## Needed Independent Val/Test Data",
        "",
        "Add a held-out validation/test pack that is not sourced from the current training clips or adjacent windows. Keep grouping by `person_id` or stable source identity.",
        "",
        "Priority cough coverage:",
        "- Dry single cough, wet cough, cough clusters, quiet cough, distant cough, cough over TV/music/speech.",
        "- Different microphones and rooms if available.",
        "- Low-SNR cough where recall is clinically important.",
        "",
        "Priority non_cough coverage:",
        "- Clean speech/conversation/narration.",
        "- Music and video audio with sudden amplitude changes.",
        "- Household noise, appliance noise, knocks, clicks, handling noise.",
        "- Silence, no-signal/static hiss, and very low RMS clips that should be excluded upstream.",
        "",
        "Priority hard negative coverage:",
        "- Sudden loud non-cough transients from music, shouting, impact, short impulse noise.",
        "- Throat clearing, sneezing, laughter, breathing/snoring/wheeze-like sounds.",
        "- FSD50K-style camera/object sounds that previously caused high baseline cough probability.",
        "",
        "## Active Learning Plan",
        "",
        "1. Run the v3 model on a larger unlabeled negative pool and rank windows by cough probability near the operating threshold.",
        "2. Prioritize review of high-confidence false-alarm candidates: high model cough probability with YAMNet/top-class evidence against cough.",
        "3. Deduplicate by source audio and time proximity so reviewers do not hear repeated adjacent windows.",
        "4. Use strata for review: top false alarms, near-threshold uncertain clips, and a small random clean control slice.",
        "5. Add only human-confirmed clips to future train; reserve a separate source-disjoint subset for val/test.",
        "6. Track false positive event types in notes using short tags instead of long prose to reduce review cost.",
        "",
        "## Next Gate",
        "",
        "Do not call this model final until the supplemental independent val/test pack shows the target recall and false-alarm rate at the intended threshold.",
        "",
    ]
    (run_dir / "eval_data_gap_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    config = read_config(args.config)
    metrics = read_json(run_dir / "metrics.json")
    dataset = read_json(args.dataset_check)
    pretrain_check = read_json(args.pretrain_check)
    pretrain_preprocess = read_json(args.pretrain_preprocess)
    pretrain_one_epoch = read_json(args.pretrain_one_epoch)
    sweep = pd.read_csv(run_dir / "threshold_sweep.csv")
    classification = read_text_if_exists(run_dir / "classification_report.txt")

    write_experiment_report(
        run_dir,
        config,
        metrics,
        dataset,
        pretrain_check,
        pretrain_preprocess,
        pretrain_one_epoch,
        sweep,
        classification,
    )
    write_board_test_plan(run_dir, config, metrics, sweep)
    write_eval_gap_report(run_dir, dataset, metrics, sweep)

    print(
        json.dumps(
            {
                "status": "completed",
                "experiment_report": str(run_dir / "experiment_report.md"),
                "board_test_plan": str(run_dir / "board_test_plan.md"),
                "eval_data_gap_report": str(run_dir / "eval_data_gap_report.md"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
