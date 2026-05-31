from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize v3.2 recall-fix sampler weights into a non-destructive label view."
    )
    parser.add_argument("--config", default="configs/audio_baseline_v3_2_recall_fix.yaml")
    parser.add_argument("--input", default="data/labels/audio_labels_v3_2_train.csv")
    parser.add_argument("--out", default="data/labels/audio_labels_v3_2_train_weighted.csv")
    parser.add_argument(
        "--report",
        default="review/v3_2_recall_fix/v3_2_sample_weight_materialization_report.json",
    )
    parser.add_argument(
        "--summary",
        default="review/v3_2_recall_fix/v3_2_sample_weight_materialization_summary.md",
    )
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
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


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return rows, list(reader.fieldnames)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def float_value(value: object, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def counter(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in Counter((row.get(column) or "").strip() for row in rows).most_common()
    }


def planned_weight(row: dict[str, str], plan: dict[str, Any]) -> tuple[float, str]:
    label = (row.get("label") or "").strip()
    stage = (row.get("source_stage") or "").strip()
    role = (row.get("candidate_role") or "").strip()
    current = float_value(row.get("sample_weight"), default=1.0)

    if label == "cough":
        if role == "positive_recall_fix_candidate":
            return float(plan.get("positive_recall_fix_candidate", 14.0)), "positive_recall_fix_candidate"
        if role == "positive_sanity_candidate":
            return float(plan.get("positive_sanity_candidate", 6.0)), "positive_sanity_candidate"
        if stage == "v3_1_fpfix_base":
            return max(current, float(plan.get("base_cough", 10.0))), "base_cough_lift"
        return max(current, float(plan.get("base_cough", 10.0))), "cough_default_lift"

    if role == "hard_negative_candidate":
        return float(plan.get("hard_negative_candidate", 4.0)), "hard_negative_candidate"
    if role == "non_cough_candidate":
        return float(plan.get("non_cough_candidate", 2.0)), "non_cough_candidate"
    if role == "clean_non_cough_smoke_candidate":
        return float(plan.get("clean_non_cough_smoke_candidate", 1.0)), "clean_non_cough_smoke_candidate"
    if stage == "v3_1_fpfix_base" and bool_value(plan.get("preserve_existing_v3_1_base_non_cough_weights")):
        return current, "preserve_v3_1_base_non_cough"
    return float(plan.get("default_non_cough", 1.0)), "default_non_cough"


def weight_summary(rows: list[dict[str, str]], weight_column: str) -> dict[str, Any]:
    total = 0.0
    label_weight = defaultdict(float)
    role_weight = defaultdict(float)
    stage_weight = defaultdict(float)
    values: list[float] = []
    for row in rows:
        weight = float_value(row.get(weight_column), default=1.0)
        values.append(weight)
        total += weight
        label_weight[(row.get("label") or "").strip()] += weight
        role_weight[(row.get("candidate_role") or "").strip()] += weight
        stage_weight[(row.get("source_stage") or "").strip()] += weight
    return {
        "total_weight": round(total, 6),
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
        "label_weight": {k: round(v, 6) for k, v in sorted(label_weight.items())},
        "source_stage_weight": {k: round(v, 6) for k, v in sorted(stage_weight.items())},
        "candidate_role_weight": {k: round(v, 6) for k, v in sorted(role_weight.items())},
        "expected_cough_sampling_fraction": round(label_weight.get("cough", 0.0) / total, 6)
        if total
        else 0.0,
    }


def simulate(rows: list[dict[str, str]], weight_column: str, batch_size: int, batches: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    weights = [float_value(row.get(weight_column), default=1.0) for row in rows]
    sampled_indices = rng.choices(range(len(rows)), weights=weights, k=batch_size * batches)
    sampled_rows = [rows[index] for index in sampled_indices]
    labels = Counter((row.get("label") or "").strip() for row in sampled_rows)
    roles = Counter((row.get("candidate_role") or "").strip() for row in sampled_rows)
    stages = Counter((row.get("source_stage") or "").strip() for row in sampled_rows)
    addon = [row for row in sampled_rows if (row.get("source_stage") or "").strip() == "v3_2_recall_fix_addon"]
    forbidden_addon = {
        "p02": sum((row.get("person_id_for_analysis") or "").strip() == "p02" for row in addon),
        "p03": sum((row.get("person_id_for_analysis") or "").strip() == "p03" for row in addon),
        "long_quiet_eval": sum((row.get("split_role_suggested") or "").strip() == "long_quiet_eval" for row in addon),
        "distance_extrap_eval": sum(
            (row.get("split_role_suggested") or "").strip() == "distance_extrap_eval" for row in addon
        ),
    }
    total = len(sampled_rows)
    return {
        "batches": batches,
        "batch_size": batch_size,
        "sampled_rows": total,
        "label_distribution": {k: int(v) for k, v in labels.most_common()},
        "observed_cough_fraction": round(labels.get("cough", 0) / max(total, 1), 6),
        "source_stage_distribution": {k: int(v) for k, v in stages.most_common()},
        "candidate_role_distribution": {k: int(v) for k, v in roles.most_common()},
        "v3_2_addon_sampled_rows": len(addon),
        "hard_negative_candidate_sampled_rows": int(roles.get("hard_negative_candidate", 0)),
        "positive_recall_fix_candidate_sampled_rows": int(roles.get("positive_recall_fix_candidate", 0)),
        "forbidden_addon_sampled_counts": forbidden_addon,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    before = report["before_weight_summary"]
    after = report["after_weight_summary"]
    sim = report["after_sampling_simulation"]
    validation = report["validation"]
    failed = [key for key, value in validation.items() if not value]
    lines = [
        "# v3.2 Sample Weight Materialization Summary",
        "",
        f"- result: `{report['result']}`",
        f"- input label: `{report['input_label_path']}`",
        f"- weighted label: `{report['weighted_label_path']}`",
        f"- rows: `{report['rows']}`",
        f"- label distribution: `{report['label_distribution']}`",
        f"- source stage distribution: `{report['source_stage_distribution']}`",
        "",
        "## Sampling Effect",
        "",
        f"- before expected cough sampling: `{before['expected_cough_sampling_fraction']}`",
        f"- after expected cough sampling: `{after['expected_cough_sampling_fraction']}`",
        f"- after observed cough fraction: `{sim['observed_cough_fraction']}`",
        f"- sampled v3.2 addon rows: `{sim['v3_2_addon_sampled_rows']}`",
        f"- sampled hard_negative_candidate rows: `{sim['hard_negative_candidate_sampled_rows']}`",
        f"- sampled positive_recall_fix_candidate rows: `{sim['positive_recall_fix_candidate_sampled_rows']}`",
        f"- forbidden addon sampled counts: `{sim['forbidden_addon_sampled_counts']}`",
        "",
        "## Policy",
        "",
        f"- target level: `{report['target_level']}`",
        f"- weight plan: `{report['sample_weight_plan']}`",
        "- loss-side class weight remains disabled; this file only materializes sampler weights.",
        "- Label-row sampling is recorded for audit; the training target is checked at feature-window manifest level during smoke.",
        "- `data/labels/audio_labels_v3_2_train.csv` was read as source and was not overwritten.",
        "",
        "## Failed Checks",
        "",
    ]
    if failed:
        lines.extend(f"- `{key}`" for key in failed)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    input_path = resolve(args.input)
    output_path = resolve(args.out)
    report_path = resolve(args.report)
    summary_path = resolve(args.summary)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output label paths must differ.")

    config = load_yaml(config_path)
    plan = dict(config.get("sampling", {}).get("sample_weight_plan") or {})
    rows, fieldnames = load_csv(input_path)
    input_hash_before = sha256_file(input_path)

    output_rows: list[dict[str, str]] = []
    materialization_reasons: Counter[str] = Counter()
    for row in rows:
        next_row = dict(row)
        original_weight = float_value(row.get("sample_weight"), default=1.0)
        new_weight, reason = planned_weight(row, plan)
        next_row["sample_weight_original"] = f"{original_weight:.6g}"
        next_row["sample_weight"] = f"{new_weight:.6g}"
        next_row["sample_weight_policy_version"] = "v3_2_weight_1"
        next_row["sample_weight_reason"] = reason
        materialization_reasons[reason] += 1
        output_rows.append(next_row)

    extra_fields = ["sample_weight_original", "sample_weight_policy_version", "sample_weight_reason"]
    output_fields = list(fieldnames)
    for field in extra_fields:
        if field not in output_fields:
            output_fields.append(field)
    write_csv(output_path, output_rows, output_fields)
    input_hash_after = sha256_file(input_path)
    output_hash = sha256_file(output_path)

    before_summary = weight_summary(rows, "sample_weight")
    after_summary = weight_summary(output_rows, "sample_weight")
    before_sim = simulate(rows, "sample_weight", args.batch_size, args.batches, args.seed)
    after_sim = simulate(output_rows, "sample_weight", args.batch_size, args.batches, args.seed)
    addon = [row for row in output_rows if (row.get("source_stage") or "").strip() == "v3_2_recall_fix_addon"]
    addon_forbidden = {
        "p02": sum((row.get("person_id_for_analysis") or "").strip() == "p02" for row in addon),
        "p03": sum((row.get("person_id_for_analysis") or "").strip() == "p03" for row in addon),
        "long_quiet_eval": sum((row.get("split_role_suggested") or "").strip() == "long_quiet_eval" for row in addon),
        "distance_extrap_eval": sum((row.get("split_role_suggested") or "").strip() == "distance_extrap_eval" for row in addon),
    }

    validation = {
        "input_label_not_overwritten": input_hash_before == input_hash_after,
        "weighted_label_created": output_path.exists(),
        "rows_2482": len(output_rows) == 2482,
        "label_distribution_unchanged": counter(output_rows, "label") == {"non_cough": 2380, "cough": 102},
        "source_stage_distribution_unchanged": counter(output_rows, "source_stage")
        == counter(rows, "source_stage"),
        "label_level_sampling_recorded_not_targeted": True,
        "v3_2_addon_sampled": after_sim["v3_2_addon_sampled_rows"] > 0,
        "hard_negative_candidate_sampled": after_sim["hard_negative_candidate_sampled_rows"] > 0,
        "positive_recall_fix_candidate_sampled": after_sim["positive_recall_fix_candidate_sampled_rows"] > 0,
        "addon_forbidden_holdouts_absent_from_weighted_label": all(value == 0 for value in addon_forbidden.values()),
        "forbidden_addon_holdouts_not_sampled": all(
            value == 0 for value in after_sim["forbidden_addon_sampled_counts"].values()
        ),
        "no_ml_data_raw_references": all("ml/data/raw" not in str(row).replace("\\", "/") for row in output_rows),
    }

    report = {
        "stage": "v3.2-WEIGHT-1",
        "config_path": str(config_path),
        "input_label_path": str(input_path),
        "weighted_label_path": str(output_path),
        "input_label_hash_before": input_hash_before,
        "input_label_hash_after": input_hash_after,
        "weighted_label_hash": output_hash,
        "rows": len(output_rows),
        "label_distribution": counter(output_rows, "label"),
        "source_stage_distribution": counter(output_rows, "source_stage"),
        "candidate_role_distribution": counter(output_rows, "candidate_role"),
        "sample_weight_plan": plan,
        "target_level": config.get("imbalance_policy", {}).get("target_level", "feature_window_manifest"),
        "sample_weight_reason_distribution": {k: int(v) for k, v in materialization_reasons.most_common()},
        "before_weight_summary": before_summary,
        "before_sampling_simulation": before_sim,
        "after_weight_summary": after_summary,
        "after_sampling_simulation": after_sim,
        "addon_forbidden_holdout_counts": addon_forbidden,
        "validation": validation,
        "result": "pass" if all(validation.values()) else "fail",
    }
    write_json(report_path, report)
    write_text(summary_path, markdown_summary(report))
    print(f"result={report['result']}")
    print(f"weighted_label={output_path}")
    print(f"expected_cough_sampling={after_summary['expected_cough_sampling_fraction']}")
    print(f"observed_cough_fraction={after_sim['observed_cough_fraction']}")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
