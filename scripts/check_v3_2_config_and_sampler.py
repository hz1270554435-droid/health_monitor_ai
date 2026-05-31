from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "review" / "v3_2_recall_fix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check v3.2 recall-fix config and run sampler dry-run without training."
    )
    parser.add_argument("--config", default="configs/audio_baseline_v3_2_recall_fix.yaml")
    parser.add_argument("--labels", default="data/labels/audio_labels_v3_2_train.csv")
    parser.add_argument("--expected-labels-csv", default="data/labels/audio_labels_v3_2_train.csv")
    parser.add_argument("--reference-final-label", default="data/labels/audio_labels_v3_2_train.csv")
    parser.add_argument(
        "--sampler-manifest",
        default=None,
        help="Optional preprocessed manifest to use for sampler dry-run. If omitted, dry-run uses label rows.",
    )
    parser.add_argument(
        "--dataset-report",
        default="review/v3_2_recall_fix/v3_2_final_train_label_dataset_check_report.json",
    )
    parser.add_argument(
        "--preprocess-report",
        default="review/v3_2_recall_fix/v3_2_final_train_label_preprocess_smoke_report.json",
    )
    parser.add_argument("--config-report", default="review/v3_2_recall_fix/v3_2_config_check_report.json")
    parser.add_argument("--config-summary", default="review/v3_2_recall_fix/v3_2_config_check_summary.md")
    parser.add_argument(
        "--sampler-report",
        default="review/v3_2_recall_fix/v3_2_sampler_dry_run_report.json",
    )
    parser.add_argument(
        "--sampler-summary",
        default="review/v3_2_recall_fix/v3_2_sampler_dry_run_summary.md",
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_path_text(value: object) -> str:
    return str(value).replace("\\", "/").strip()


def float_value(value: object, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def counters(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in Counter((row.get(column) or "").strip() for row in rows).most_common()
    }


def read_source(path: str) -> str:
    full_path = ROOT / path
    return full_path.read_text(encoding="utf-8") if full_path.exists() else ""


def inspect_code_support() -> dict[str, Any]:
    sampling_py = read_source("src/audio/sampling.py")
    train_py = read_source("scripts/train_audio_baseline.py")
    preprocess_py = read_source("scripts/preprocess_audio.py")
    features_py = read_source("src/audio/features.py")
    return {
        "weighted_sampler_supported": "WeightedRandomSampler" in sampling_py
        and "build_weighted_sampler" in train_py,
        "weighted_random_sample_weight_supported": "row_sample_weights" in sampling_py
        and "sampling.mode=weighted_random" in sampling_py
        and '"sample_weight"' in preprocess_py,
        "balanced_sampler_supported": '"balanced"' in sampling_py,
        "hard_negative_oversample_supported": "hard_negative_oversample" in sampling_py,
        "source_weighted_supported": '"source_weighted"' in sampling_py,
        "source_stage_candidate_role_weighting_supported": "source_stage" in sampling_py
        or "candidate_role" in sampling_py,
        "loss_class_weight_helper_exists": "def loss_class_weights" in sampling_py,
        "loss_class_weight_consumed_by_train_script": "loss_class_weights" in train_py
        or "CrossEntropyLoss(weight=" in train_py,
        "plain_cross_entropy_loss": "CrossEntropyLoss()" in train_py,
        "preprocess_copies_sample_weight": '"sample_weight"' in preprocess_py,
        "audio_loader_mono_true": "mono=True" in features_py,
        "train_reads_features_not_audio": "FeatureDataset" in train_py and "load_audio_segment" not in train_py,
    }


def proposed_policy_weight(row: dict[str, str], plan: dict[str, Any]) -> float:
    label = (row.get("label") or "").strip()
    stage = (row.get("source_stage") or "").strip()
    role = (row.get("candidate_role") or "").strip()
    current = float_value(row.get("sample_weight"), default=1.0)
    if label == "cough":
        if role == "positive_recall_fix_candidate":
            return float(plan.get("positive_recall_fix_candidate", 14.0))
        if role == "positive_sanity_candidate":
            return float(plan.get("positive_sanity_candidate", 6.0))
        if stage == "v3_1_fpfix_base":
            return max(current, float(plan.get("base_cough", 10.0)))
        return max(current, float(plan.get("base_cough", 10.0)))
    if role == "hard_negative_candidate":
        return float(plan.get("hard_negative_candidate", 4.0))
    if role == "non_cough_candidate":
        return float(plan.get("non_cough_candidate", 2.0))
    if role == "clean_non_cough_smoke_candidate":
        return float(plan.get("clean_non_cough_smoke_candidate", 1.0))
    if stage == "v3_1_fpfix_base" and bool_value(plan.get("preserve_existing_v3_1_base_non_cough_weights")):
        return current
    return float(plan.get("default_non_cough", 1.0))


def summarize_weights(rows: list[dict[str, str]], weights: list[float]) -> dict[str, Any]:
    total_weight = sum(weights)
    label_weight = defaultdict(float)
    role_weight = defaultdict(float)
    stage_weight = defaultdict(float)
    for row, weight in zip(rows, weights):
        label_weight[(row.get("label") or "").strip()] += weight
        role_weight[(row.get("candidate_role") or "").strip()] += weight
        stage_weight[(row.get("source_stage") or "").strip()] += weight
    return {
        "total_weight": round(total_weight, 6),
        "label_weight": {k: round(v, 6) for k, v in sorted(label_weight.items())},
        "source_stage_weight": {k: round(v, 6) for k, v in sorted(stage_weight.items())},
        "candidate_role_weight": {k: round(v, 6) for k, v in sorted(role_weight.items())},
        "expected_positive_fraction": round(label_weight.get("cough", 0.0) / total_weight, 6)
        if total_weight > 0
        else 0.0,
    }


def simulate_batches(
    rows: list[dict[str, str]],
    weights: list[float],
    batch_size: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    population = list(range(len(rows)))
    sampled = rng.choices(population, weights=weights, k=batch_size * batches)
    label_counts = Counter((rows[index].get("label") or "").strip() for index in sampled)
    role_counts = Counter((rows[index].get("candidate_role") or "").strip() for index in sampled)
    stage_counts = Counter((rows[index].get("source_stage") or "").strip() for index in sampled)
    addon_rows = [rows[index] for index in sampled if (rows[index].get("source_stage") or "").strip() == "v3_2_recall_fix_addon"]
    forbidden_addon_counts = {
        "p02": sum((row.get("person_id_for_analysis") or "").strip() == "p02" for row in addon_rows),
        "p03": sum((row.get("person_id_for_analysis") or "").strip() == "p03" for row in addon_rows),
        "long_quiet_eval": sum((row.get("split_role_suggested") or "").strip() == "long_quiet_eval" for row in addon_rows),
        "distance_extrap_eval": sum(
            (row.get("split_role_suggested") or "").strip() == "distance_extrap_eval" for row in addon_rows
        ),
    }
    total = len(sampled)
    return {
        "batches": batches,
        "batch_size": batch_size,
        "sampled_rows": total,
        "sampled_label_distribution": {k: int(v) for k, v in label_counts.most_common()},
        "sampled_cough_fraction": round(label_counts.get("cough", 0) / max(total, 1), 6),
        "sampled_source_stage_distribution": {k: int(v) for k, v in stage_counts.most_common()},
        "sampled_candidate_role_distribution": {k: int(v) for k, v in role_counts.most_common()},
        "sampled_v3_2_addon_rows": len(addon_rows),
        "sampled_forbidden_addon_counts": forbidden_addon_counts,
    }


def build_config_report(
    config: dict[str, Any],
    labels: list[dict[str, str]],
    dataset_report: dict[str, Any],
    preprocess_report: dict[str, Any],
    config_path: Path,
    labels_path: Path,
    expected_labels_csv: str,
    reference_final_label_path: Path,
) -> dict[str, Any]:
    paths = config.get("paths", {})
    audio = config.get("audio", {})
    labels_cfg = config.get("labels", {})
    sampling = config.get("sampling", {})
    code_support = inspect_code_support()
    label_distribution = counters(labels, "label")
    source_stage_distribution = counters(labels, "source_stage")
    addon = [row for row in labels if (row.get("source_stage") or "").strip() == "v3_2_recall_fix_addon"]
    reference_final_label_hash = sha256_file(reference_final_label_path) if reference_final_label_path.exists() else None
    old_label_paths = {
        "data/labels/audio_labels_v3.csv": ROOT / "data" / "labels" / "audio_labels_v3.csv",
        "data/labels/audio_labels_v3_1_fpfix_train.csv": ROOT
        / "data"
        / "labels"
        / "audio_labels_v3_1_fpfix_train.csv",
    }
    current_old_label_hashes = {
        key: sha256_file(path) for key, path in old_label_paths.items() if path.exists()
    }
    expected_old_label_hashes = dict(dataset_report.get("old_label_hash_after") or {})
    addon_forbidden = {
        "p02": sum((row.get("person_id_for_analysis") or "").strip() == "p02" for row in addon),
        "p03": sum((row.get("person_id_for_analysis") or "").strip() == "p03" for row in addon),
        "long_quiet_eval": sum((row.get("split_role_suggested") or "").strip() == "long_quiet_eval" for row in addon),
        "distance_extrap_eval": sum((row.get("split_role_suggested") or "").strip() == "distance_extrap_eval" for row in addon),
    }
    checks = {
        "config_exists": config_path.exists(),
        "labels_path_matches_expected": norm_path_text(paths.get("labels_csv"))
        == norm_path_text(expected_labels_csv),
        "labels_file_exists": labels_path.exists(),
        "labels_arg_matches_expected": norm_path_text(labels_path.relative_to(ROOT) if labels_path.is_relative_to(ROOT) else labels_path)
        == norm_path_text(expected_labels_csv),
        "label_rows_2482": len(labels) == 2482,
        "label_distribution_matches_expected": label_distribution == {"non_cough": 2380, "cough": 102},
        "run_name_matches": config.get("run", {}).get("name") == "audio_baseline_v3_2_recall_fix",
        "frontend_board_htk_no_norm_v1": audio.get("frontend") == "board_htk_no_norm_v1",
        "sample_rate_16000": int(audio.get("sample_rate", 0)) == 16000,
        "mono_true_configured_or_loader_confirmed": bool_value(audio.get("mono", True))
        and code_support["audio_loader_mono_true"],
        "window_seconds_1_0": float(audio.get("window_seconds", 0)) == 1.0,
        "hop_seconds_0_5": float(audio.get("hop_seconds", 0)) == 0.5,
        "n_mels_40": int(audio.get("n_mels", 0)) == 40,
        "mel_scale_htk": str(audio.get("mel_scale", "")).lower() == "htk",
        "normalization_none": str(audio.get("normalize", "")).lower() == "none",
        "class_order_non_cough_cough": labels_cfg.get("class_order") == ["non_cough", "cough"],
        "processed_dir_v3_2_dedicated": "v3_2_recall_fix" in norm_path_text(paths.get("processed_dir"))
        and norm_path_text(paths.get("processed_dir")) != "data/processed/audio_features_v3_1_fpfix",
        "models_dir_v3_2_dedicated": "v3_2_recall_fix" in norm_path_text(paths.get("models_dir"))
        and norm_path_text(paths.get("models_dir")) != "models/audio",
        "results_dir_v3_2_dedicated": "v3_2_recall_fix" in norm_path_text(paths.get("results_dir"))
        and norm_path_text(paths.get("results_dir")) != "results/audio",
        "legacy_dual_channel_caveat_recorded": bool(
            config.get("imbalance_policy", {}).get("legacy_dual_channel_caveat", {}).get("accepted")
        )
        and bool(dataset_report.get("legacy_base_channel_caveat")),
        "dataset_report_pass_or_caveat": dataset_report.get("result") in {"pass", "pass_with_legacy_base_channel_caveat"},
        "preprocess_smoke_pass": preprocess_report.get("result") == "pass",
        "training_entry_uses_mono_loader": code_support["train_reads_features_not_audio"]
        and code_support["audio_loader_mono_true"],
        "weighted_sampler_supported": code_support["weighted_sampler_supported"],
        "sample_weight_column_supported": code_support["weighted_random_sample_weight_supported"],
        "class_weight_not_claimed_active": config.get("training", {}).get("class_weight") in {None, "none", False},
        "source_stage_candidate_role_rules_marked_not_direct": not bool(
            config.get("imbalance_policy", {})
            .get("active_without_training_code_patch", {})
            .get("source_stage_candidate_role_rules")
        ),
        "addon_forbidden_holdouts_zero": all(value == 0 for value in addon_forbidden.values()),
        "no_ml_data_raw_references": all("ml/data/raw" not in norm_path_text(row) for row in labels),
        "old_label_hashes_match_final_check": all(
            current_old_label_hashes.get(key) == value for key, value in expected_old_label_hashes.items()
        ),
        "reference_final_label_exists": reference_final_label_path.exists(),
    }
    return {
        "stage": "v3.2-CONFIG-1A-CONFIG-CHECK",
        "config_path": str(config_path),
        "label_path": str(labels_path),
        "label_hash": sha256_file(labels_path),
        "expected_labels_csv": expected_labels_csv,
        "reference_final_label_path": str(reference_final_label_path),
        "reference_final_label_hash": reference_final_label_hash,
        "rows": len(labels),
        "label_distribution": label_distribution,
        "source_stage_distribution": source_stage_distribution,
        "addon_forbidden_holdout_counts": addon_forbidden,
        "output_dirs": {
            "processed_dir": paths.get("processed_dir"),
            "models_dir": paths.get("models_dir"),
            "results_dir": paths.get("results_dir"),
        },
        "frontend": {
            "frontend": audio.get("frontend"),
            "sample_rate": audio.get("sample_rate"),
            "mono": audio.get("mono"),
            "window_seconds": audio.get("window_seconds"),
            "hop_seconds": audio.get("hop_seconds"),
            "n_mels": audio.get("n_mels"),
            "mel_scale": audio.get("mel_scale"),
            "normalize": audio.get("normalize"),
            "class_order": labels_cfg.get("class_order"),
        },
        "sampler_config": sampling,
        "code_support": code_support,
        "dataset_report_result": dataset_report.get("result"),
        "preprocess_smoke_result": preprocess_report.get("result"),
        "legacy_dual_channel_caveat": {
            "accepted": bool(dataset_report.get("legacy_base_channel_caveat")),
            "channels_counts_by_stage": dataset_report.get("channels_counts_by_stage", {}),
            "config_recorded": config.get("imbalance_policy", {}).get("legacy_dual_channel_caveat", {}),
        },
        "old_label_hashes": {
            "current": current_old_label_hashes,
            "expected_from_final_check": expected_old_label_hashes,
        },
        "validation": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }


def build_sampler_report(
    config: dict[str, Any],
    labels: list[dict[str, str]],
    batches: int,
    batch_size: int,
    seed: int,
    sampler_input_path: Path | None = None,
    sampler_input_level: str = "label_rows",
) -> dict[str, Any]:
    sampling = config.get("sampling", {})
    plan = dict(sampling.get("sample_weight_plan") or {})
    current_weights = [float_value(row.get("sample_weight"), default=1.0) for row in labels]
    proposed_weights = [proposed_policy_weight(row, plan) for row in labels]
    low, high = sampling.get("target_positive_fraction_range", [0.25, 0.35])
    target_low = float(low)
    target_high = float(high)
    current_summary = summarize_weights(labels, current_weights)
    proposed_summary = summarize_weights(labels, proposed_weights)
    current_sim = simulate_batches(labels, current_weights, batch_size=batch_size, batches=batches, seed=seed)
    proposed_sim = simulate_batches(labels, proposed_weights, batch_size=batch_size, batches=batches, seed=seed)
    current_in_range = target_low <= current_summary["expected_positive_fraction"] <= target_high
    proposed_in_range = target_low <= proposed_summary["expected_positive_fraction"] <= target_high
    validation = {
        "weighted_random_configured": str(sampling.get("mode", "")).lower() == "weighted_random",
        "sample_weight_column_configured": bool_value(sampling.get("use_sample_weight")),
        "current_manifest_sample_weight_numeric_positive": all(weight > 0 for weight in current_weights),
        "current_sample_weight_hits_target_positive_fraction": current_in_range,
        "proposed_materialized_policy_hits_target_positive_fraction": proposed_in_range,
        "proposed_materialized_policy_samples_v3_2_addon": proposed_sim["sampled_v3_2_addon_rows"] > 0,
        "proposed_materialized_policy_keeps_forbidden_addon_holdouts_out": all(
            value == 0 for value in proposed_sim["sampled_forbidden_addon_counts"].values()
        ),
        "training_not_run": True,
    }
    result = "pass_requires_sample_weight_materialization" if proposed_in_range and not current_in_range else (
        "pass" if current_in_range else "needs_policy_adjustment"
    )
    return {
        "stage": "v3.2-CONFIG-1A-SAMPLER-DRY-RUN",
        "sampler_input_path": str(sampler_input_path) if sampler_input_path else None,
        "sampler_input_level": sampler_input_level,
        "rows": len(labels),
        "batch_size": batch_size,
        "batches": batches,
        "seed": seed,
        "target_positive_fraction_range": [target_low, target_high],
        "current_manifest_sample_weight": {
            "summary": current_summary,
            "simulation": current_sim,
            "status": "active_if_preprocessed_manifest_preserves_sample_weight",
        },
        "proposed_materialized_weight_policy": {
            "sample_weight_plan": plan,
            "summary": proposed_summary,
            "simulation": proposed_sim,
            "status": "dry_run_only_until_weights_are_materialized",
        },
        "validation": validation,
        "result": result,
    }


def md_config_summary(report: dict[str, Any]) -> str:
    validation = report["validation"]
    failed = [key for key, value in validation.items() if not value]
    lines = [
        "# v3.2 Config Check Summary",
        "",
        f"- result: `{report['result']}`",
        f"- config: `{report['config_path']}`",
        f"- labels: `{report['label_path']}`",
        f"- rows: `{report['rows']}`",
        f"- label distribution: `{report['label_distribution']}`",
        f"- output dirs: `{report['output_dirs']}`",
        f"- frontend/class order: `{report['frontend']}`",
        f"- dataset report result: `{report['dataset_report_result']}`",
        f"- preprocess smoke result: `{report['preprocess_smoke_result']}`",
        "",
        "## Code Support",
        "",
    ]
    for key, value in report["code_support"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Legacy Dual-channel Caveat",
            "",
            "- Accepted for preserved v3.1 base only; training/preprocess must keep `mono=True` or equivalent mono conversion.",
            f"- channels by stage: `{report['legacy_dual_channel_caveat']['channels_counts_by_stage']}`",
            "",
            "## Failed Checks",
            "",
        ]
    )
    if failed:
        lines.extend(f"- `{key}`" for key in failed)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def md_sampler_summary(report: dict[str, Any]) -> str:
    current = report["current_manifest_sample_weight"]
    proposed = report["proposed_materialized_weight_policy"]
    validation = report["validation"]
    lines = [
        "# v3.2 Sampler Dry-run Summary",
        "",
        f"- result: `{report['result']}`",
        f"- rows: `{report['rows']}`",
        f"- batches: `{report['batches']}`",
        f"- batch size: `{report['batch_size']}`",
        f"- target positive fraction range: `{report['target_positive_fraction_range']}`",
        "",
        "## Current Manifest `sample_weight`",
        "",
        f"- expected positive fraction: `{current['summary']['expected_positive_fraction']}`",
        f"- simulated cough fraction: `{current['simulation']['sampled_cough_fraction']}`",
        f"- sampled label distribution: `{current['simulation']['sampled_label_distribution']}`",
        f"- sampled v3.2 addon rows: `{current['simulation']['sampled_v3_2_addon_rows']}`",
        "",
        "## Proposed Materialized Policy",
        "",
        f"- expected positive fraction: `{proposed['summary']['expected_positive_fraction']}`",
        f"- simulated cough fraction: `{proposed['simulation']['sampled_cough_fraction']}`",
        f"- sampled label distribution: `{proposed['simulation']['sampled_label_distribution']}`",
        f"- sampled v3.2 addon rows: `{proposed['simulation']['sampled_v3_2_addon_rows']}`",
        f"- forbidden addon sampled counts: `{proposed['simulation']['sampled_forbidden_addon_counts']}`",
        "",
        "## Interpretation",
        "",
        "- The sampler path is supported, but the current final label `sample_weight` column still reflects the v3.1-fpfix/base weights and does not by itself solve the v3.2 4.1% cough imbalance.",
        "- The proposed policy reaches the 25%-35% cough dry-run target only after the policy weights are materialized into the manifest/label sample_weight used by preprocessing.",
        "- No training, one-epoch smoke, feature generation, raw-data write, or label merge was run.",
        "",
        "## Validation",
        "",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    labels_path = resolve(args.labels)
    config = load_yaml(config_path)
    labels = load_csv(labels_path)
    sampler_input_path = resolve(args.sampler_manifest) if args.sampler_manifest else None
    sampler_rows = load_csv(sampler_input_path) if sampler_input_path and sampler_input_path.exists() else labels
    sampler_input_level = "feature_window_manifest" if sampler_input_path and sampler_input_path.exists() else "label_rows"
    dataset_report = load_json(resolve(args.dataset_report))
    preprocess_report = load_json(resolve(args.preprocess_report))
    batch_size = int(args.batch_size or config.get("training", {}).get("batch_size", 32))

    config_report = build_config_report(
        config=config,
        labels=labels,
        dataset_report=dataset_report,
        preprocess_report=preprocess_report,
        config_path=config_path,
        labels_path=labels_path,
        expected_labels_csv=args.expected_labels_csv,
        reference_final_label_path=resolve(args.reference_final_label),
    )
    sampler_report = build_sampler_report(
        config=config,
        labels=sampler_rows,
        batches=int(args.batches),
        batch_size=batch_size,
        seed=int(args.seed),
        sampler_input_path=sampler_input_path,
        sampler_input_level=sampler_input_level,
    )

    write_json(config_report, resolve(args.config_report))
    write_text(md_config_summary(config_report), resolve(args.config_summary))
    write_json(sampler_report, resolve(args.sampler_report))
    write_text(md_sampler_summary(sampler_report), resolve(args.sampler_summary))

    print(f"config_check_result={config_report['result']}")
    print(f"sampler_dry_run_result={sampler_report['result']}")
    return 0 if config_report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
