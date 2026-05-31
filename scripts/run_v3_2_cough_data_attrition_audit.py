from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v3.2 cough data attrition from raw folders to final train label.")
    parser.add_argument("--window-manifest", default="review/v3_2_recall_fix/window_manifest_combined.csv")
    parser.add_argument("--review-plan", default="review/v3_2_recall_fix/review_plan_combined_corrected.csv")
    parser.add_argument(
        "--review-plan-applied",
        default="review/v3_2_recall_fix/review_plan_combined_corrected_qa_recheck_applied.csv",
    )
    parser.add_argument(
        "--candidate-csv",
        default="review/v3_2_recall_fix/audio_labels_v3_2_candidate_recall_fix.csv",
    )
    parser.add_argument(
        "--merge-dry-run",
        default="review/v3_2_recall_fix/v3_2_candidate_merge_dry_run.csv",
    )
    parser.add_argument("--final-train", default="data/labels/audio_labels_v3_2_train.csv")
    parser.add_argument(
        "--report-csv",
        default="review/v3_2_recall_fix/v3_2_cough_data_attrition_report.csv",
    )
    parser.add_argument(
        "--summary-md",
        default="review/v3_2_recall_fix/v3_2_cough_data_attrition_summary.md",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def folder_key(row: dict[str, str]) -> str:
    return (row.get("folder_name") or "").strip()


def is_cough_row(row: dict[str, str]) -> bool:
    return (row.get("scenario_canonical") or "").strip() == "cough"


def summarize_attrition(
    window_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
    applied_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    merge_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cough_windows = [row for row in window_rows if is_cough_row(row)]
    review_cough = [row for row in review_rows if (row.get("scenario_canonical") or "").strip() == "cough"]
    applied_cough = [row for row in applied_rows if (row.get("scenario_canonical") or "").strip() == "cough"]
    candidate_cough = [row for row in candidate_rows if (row.get("label") or "").strip() == "cough"]
    merge_cough = [row for row in merge_rows if (row.get("label") or "").strip() == "cough"]
    final_addon_cough = [
        row for row in final_rows
        if (row.get("source_stage") or "").strip() == "v3_2_recall_fix_addon" and (row.get("label") or "").strip() == "cough"
    ]

    folders = sorted({folder_key(row) for row in cough_windows if folder_key(row)})
    rows_out: list[dict[str, Any]] = []

    for folder in folders:
        manifest_rows = [row for row in cough_windows if folder_key(row) == folder]
        review_folder = [row for row in review_cough if folder_key(row) == folder]
        applied_folder = [row for row in applied_cough if folder_key(row) == folder]
        candidate_folder = [row for row in candidate_cough if folder_key(row) == folder]
        merge_folder = [row for row in merge_cough if folder_key(row) == folder]
        final_folder = [row for row in final_addon_cough if folder_key(row) == folder]

        wavs = {row.get("audio_file", "") for row in manifest_rows if row.get("audio_file")}
        duration_sum = sum(float_value(row.get("duration_s")) for row in {tuple(sorted(r.items())): r for r in manifest_rows}.values())

        holdout_distance = sum(
            1 for row in merge_folder if (row.get("proposed_merge_role") or "").strip() == "eval_distance_extrap"
        )
        holdout_unseen = sum(
            1 for row in merge_folder if (row.get("proposed_merge_role") or "").strip() == "eval_unseen_person"
        )
        holdout_sanity = sum(
            1 for row in merge_folder if (row.get("proposed_merge_role") or "").strip() == "eval_sanity"
        )
        holdout_other = sum(
            1
            for row in merge_folder
            if (row.get("proposed_merge_role") or "").strip() not in {"train_addon", "eval_distance_extrap", "eval_unseen_person", "eval_sanity"}
        )
        train_addon_cough = sum(
            1 for row in merge_folder if (row.get("proposed_merge_role") or "").strip() == "train_addon"
        )
        unreviewed_high_value = max(len(manifest_rows) - len(review_folder), 0)

        persons = sorted({(row.get("person_id") or row.get("person_id_for_analysis") or "").strip() for row in manifest_rows if (row.get("person_id") or row.get("person_id_for_analysis") or "").strip()})
        split_roles = sorted({(row.get("split_role_suggested") or "").strip() for row in manifest_rows if (row.get("split_role_suggested") or "").strip()})

        rows_out.append(
            {
                "folder_name": folder,
                "person_ids": ",".join(persons),
                "split_roles": ",".join(split_roles),
                "raw_wav_count": len(wavs),
                "total_duration_sec": round(duration_sum, 3),
                "raw_window_count": len(manifest_rows),
                "review_plan_event_count": len(review_folder),
                "manual_reviewed_event_count": len(applied_folder),
                "candidate_cough_count": len(candidate_folder),
                "merge_dry_run_cough_rows": len(merge_folder),
                "final_train_addon_cough_count": len(final_folder),
                "excluded_eval_sanity_count": holdout_sanity,
                "excluded_unseen_person_count": holdout_unseen,
                "excluded_distance_extrap_count": holdout_distance,
                "excluded_other_holdout_count": holdout_other,
                "train_addon_cough_count": train_addon_cough,
                "unreviewed_high_value_window_estimate": unreviewed_high_value,
            }
        )

    p01_train = sum(1 for row in final_addon_cough if (row.get("person_id_for_analysis") or "").strip() == "p01")
    p02_sanity = sum(
        1 for row in merge_cough if (row.get("person_id_for_analysis") or "").strip() == "p02" and (row.get("proposed_merge_role") or "").strip() == "eval_sanity"
    )
    p03_unseen = sum(
        1 for row in merge_cough if (row.get("person_id_for_analysis") or "").strip() == "p03" and (row.get("proposed_merge_role") or "").strip() == "eval_unseen_person"
    )
    distance_holdout = sum(
        1
        for row in merge_cough
        if (row.get("proposed_merge_role") or "").strip() == "eval_distance_extrap"
        and any(token in (row.get("source_audio_path") or row.get("source_audio") or row.get("audio_file") or row.get("folder_name") or "") for token in ["70cm", "100cm"])
    )

    top_unreviewed = sorted(rows_out, key=lambda item: item["unreviewed_high_value_window_estimate"], reverse=True)[:5]
    summary = {
        "raw_cough_window_count": len(cough_windows),
        "reviewed_cough_event_count": len(applied_cough),
        "candidate_cough_count": len(candidate_cough),
        "final_train_cough_count": len(final_addon_cough),
        "p01_final_train_addon_cough": p01_train,
        "p02_eval_sanity_excluded_cough": p02_sanity,
        "p03_unseen_holdout_excluded_cough": p03_unseen,
        "distance_extrap_excluded_cough_70cm_100cm": distance_holdout,
        "top_unreviewed_folders": top_unreviewed,
    }
    return rows_out, summary


def markdown_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v3.2 Cough Data Attrition Summary",
        "",
        f"- raw cough windows: `{summary['raw_cough_window_count']}`",
        f"- reviewed cough events: `{summary['reviewed_cough_event_count']}`",
        f"- candidate cough rows: `{summary['candidate_cough_count']}`",
        f"- final train addon cough rows: `{summary['final_train_cough_count']}`",
        "",
        "## Person-level highlights",
        "",
        f"- p01 cough entering final train addon: `{summary['p01_final_train_addon_cough']}`",
        f"- p02 cough excluded as eval_sanity: `{summary['p02_eval_sanity_excluded_cough']}`",
        f"- p03 cough excluded as unseen holdout: `{summary['p03_unseen_holdout_excluded_cough']}`",
        f"- 70cm/100cm cough excluded as distance extrap: `{summary['distance_extrap_excluded_cough_70cm_100cm']}`",
        "",
        "## Folder attrition overview",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['folder_name']}` raw_windows `{row['raw_window_count']}`, review_events `{row['review_plan_event_count']}`, "
            f"candidate_cough `{row['candidate_cough_count']}`, final_train_addon_cough `{row['final_train_addon_cough_count']}`, "
            f"holdout(sanity/unseen/distance) `{row['excluded_eval_sanity_count']}/{row['excluded_unseen_person_count']}/{row['excluded_distance_extrap_count']}`, "
            f"unreviewed_high_value_est `{row['unreviewed_high_value_window_estimate']}`"
        )
    lines.extend([
        "",
        "## Recommended remediation",
        "",
        "- Re-draw review clips from folders with the highest unreviewed cough window estimates first.",
        "- Prioritize p01 train_candidate folders if the goal is to add more positive_recall_fix_candidate without breaking unseen/distance holds.",
        "- Avoid consuming p03 unseen-person folders or 70cm/100cm distance-extrap folders into train unless the evaluation policy is explicitly changed.",
        "- If more positives are needed without breaking holdouts, prioritize remaining unreviewed windows in p01 40cm / near-field folders and any p02 material only if it stays eval_sanity-only.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    window_rows = load_csv(resolve(args.window_manifest))
    review_rows = load_csv(resolve(args.review_plan))
    applied_rows = load_csv(resolve(args.review_plan_applied))
    candidate_rows = load_csv(resolve(args.candidate_csv))
    merge_rows = load_csv(resolve(args.merge_dry_run))
    final_rows = load_csv(resolve(args.final_train))

    rows_out, summary = summarize_attrition(window_rows, review_rows, applied_rows, candidate_rows, merge_rows, final_rows)
    fieldnames = list(rows_out[0].keys()) if rows_out else [
        "folder_name",
        "person_ids",
        "split_roles",
        "raw_wav_count",
        "total_duration_sec",
        "raw_window_count",
        "review_plan_event_count",
        "manual_reviewed_event_count",
        "candidate_cough_count",
        "merge_dry_run_cough_rows",
        "final_train_addon_cough_count",
        "excluded_eval_sanity_count",
        "excluded_unseen_person_count",
        "excluded_distance_extrap_count",
        "excluded_other_holdout_count",
        "train_addon_cough_count",
        "unreviewed_high_value_window_estimate",
    ]
    write_csv(resolve(args.report_csv), rows_out, fieldnames)
    write_text(resolve(args.summary_md), markdown_summary(summary, rows_out))
    print(json.dumps({"status": "passed", **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
