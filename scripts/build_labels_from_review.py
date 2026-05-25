from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.io import ensure_dir


MANUAL_MAP = {
    "cough": {"label": "cough", "hard_negative": False, "sample_weight": 1.0, "source": "yamnet_review_gold"},
    "hard_negative": {
        "label": "non_cough",
        "hard_negative": True,
        "sample_weight": 1.0,
        "source": "yamnet_review_gold",
    },
    "clean_non_cough": {
        "label": "non_cough",
        "hard_negative": False,
        "sample_weight": 1.0,
        "source": "yamnet_review_gold",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a new audio labels CSV from a reviewed YAMNet sheet.")
    parser.add_argument("--base-labels", required=True)
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--include-pseudo", action="store_true")
    parser.add_argument("--pseudo-cough-weight", type=float, default=0.5)
    parser.add_argument("--pseudo-non-cough-weight", type=float, default=0.3)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def stable_id(prefix: str, source_audio: str, start_time: float, end_time: float, decision: str) -> str:
    digest = hashlib.sha1(f"{source_audio}:{start_time:.6f}:{end_time:.6f}:{decision}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_manual(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def pseudo_mapping(row: pd.Series, args: argparse.Namespace) -> dict[str, object] | None:
    if not args.include_pseudo:
        return None
    decision = str(row.get("decision", "")).strip()
    if decision == "pseudo_cough_candidate":
        return {
            "label": "cough",
            "hard_negative": False,
            "sample_weight": float(args.pseudo_cough_weight),
            "source": "yamnet_pseudo_label",
        }
    if decision == "pseudo_non_cough_candidate":
        return {
            "label": "non_cough",
            "hard_negative": False,
            "sample_weight": float(args.pseudo_non_cough_weight),
            "source": "yamnet_pseudo_label",
        }
    return None


def make_label_row(row: pd.Series, mapping: dict[str, object], output_columns: list[str]) -> dict[str, object]:
    source_audio = str(row["source_audio"])
    start_time = float(row["start_time"])
    end_time = float(row["end_time"])
    decision = str(row.get("manual_decision") or row.get("decision") or "review")
    clip_id = stable_id("yamnet_review", source_audio, start_time, end_time, decision)
    person_id = stable_id("P_YAMNET", source_audio, 0.0, 0.0, "source")[:22]

    values = {column: "" for column in output_columns}
    values.update(
        {
            "clip_id": clip_id,
            "session_id": f"session_{clip_id}",
            "person_id": person_id,
            "start_time": start_time,
            "end_time": end_time,
            "label": mapping["label"],
            "audio_file": source_audio,
            "hard_negative": bool(mapping["hard_negative"]),
            "sample_weight": float(mapping["sample_weight"]),
            "source": mapping["source"],
            "change_reason": str(mapping["source"]),
        }
    )
    if "split" in output_columns:
        values["split"] = "train"
    return values


def write_report(
    report_path: Path,
    base_rows: int,
    added_rows: pd.DataFrame,
    skipped_counts: dict[str, int],
    include_pseudo: bool,
    out_labels: Path,
) -> None:
    ensure_dir(report_path.parent)
    lines = [
        "# audio_labels_v3 from YAMNet review",
        "",
        "This report documents label generation from human-reviewed YAMNet hard-mining clips.",
        "The base labels file was not overwritten.",
        "",
        f"- Base rows: {base_rows}",
        f"- Added rows: {len(added_rows)}",
        f"- Include pseudo labels: {include_pseudo}",
        f"- Output labels: `{out_labels}`",
        "",
        "## Added label counts",
        "",
    ]
    if added_rows.empty:
        lines.append("_None._")
    else:
        for label, count in added_rows["label"].value_counts().sort_index().items():
            lines.append(f"- {label}: {int(count)}")
    lines.extend(["", "## Added source counts", ""])
    if added_rows.empty or "source" not in added_rows.columns:
        lines.append("_None._")
    else:
        for source, count in added_rows["source"].value_counts().sort_index().items():
            lines.append(f"- {source}: {int(count)}")
    lines.extend(["", "## Skipped decisions", ""])
    if not skipped_counts:
        lines.append("_None._")
    else:
        for decision, count in sorted(skipped_counts.items()):
            lines.append(f"- {decision}: {count}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    base_labels_path = resolve_path(args.base_labels)
    review_sheet_path = resolve_path(args.review_sheet)
    out_labels_path = resolve_path(args.out_labels)
    out_report_path = resolve_path(args.out_report)

    if not base_labels_path.exists():
        raise FileNotFoundError(f"Base labels not found: {base_labels_path}")
    if not review_sheet_path.exists():
        raise FileNotFoundError(f"Review sheet not found: {review_sheet_path}")
    if out_labels_path.resolve() == base_labels_path.resolve():
        raise ValueError("--out-labels must not overwrite --base-labels")

    base_labels = pd.read_csv(base_labels_path)
    review = pd.read_csv(review_sheet_path)
    output_columns = list(base_labels.columns)
    for extra in ("sample_weight", "source"):
        if extra not in output_columns:
            output_columns.append(extra)

    for column in output_columns:
        if column not in base_labels.columns:
            base_labels[column] = ""

    new_rows: list[dict[str, object]] = []
    skipped_counts: dict[str, int] = {}
    for _, row in review.iterrows():
        manual = normalize_manual(row.get("manual_decision", ""))
        if manual in ("uncertain", "exclude"):
            skipped_counts[manual] = skipped_counts.get(manual, 0) + 1
            continue
        mapping = MANUAL_MAP.get(manual)
        if mapping is None and manual:
            skipped_counts[f"unsupported_manual:{manual}"] = skipped_counts.get(f"unsupported_manual:{manual}", 0) + 1
            continue
        if mapping is None:
            mapping = pseudo_mapping(row, args)
        if mapping is None:
            decision = str(row.get("decision", "unreviewed"))
            skipped_counts[decision] = skipped_counts.get(decision, 0) + 1
            continue
        new_rows.append(make_label_row(row, mapping, output_columns))

    added = pd.DataFrame(new_rows, columns=output_columns)
    output = pd.concat([base_labels[output_columns], added], ignore_index=True)
    ensure_dir(out_labels_path.parent)
    output.to_csv(out_labels_path, index=False)
    write_report(out_report_path, len(base_labels), added, skipped_counts, args.include_pseudo, out_labels_path)
    print(
        f"OK: wrote {len(output)} rows to {out_labels_path}; "
        f"added={len(added)}; include_pseudo={args.include_pseudo}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
