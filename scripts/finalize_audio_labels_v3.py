from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


ALLOWED_LABELS = {"cough", "non_cough"}
ALLOWED_SPLITS = {"train", "val", "test"}
PROTECTED_COLUMNS = ["audio_file", "start_time", "end_time", "label", "split", "person_id"]
REQUIRED_COLUMNS = {
    "clip_id",
    "session_id",
    "person_id",
    "start_time",
    "end_time",
    "label",
    "audio_file",
    "split",
    "sample_weight",
    "hard_negative",
    "hard_negative_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize audio_labels_v3.csv from validated audio_labels_v3_candidate.csv."
    )
    parser.add_argument("--candidate", default="data/labels/audio_labels_v3_candidate.csv")
    parser.add_argument("--out-labels", default="data/labels/audio_labels_v3.csv")
    parser.add_argument("--base-labels", default="data/labels/audio_labels_v2.csv")
    parser.add_argument(
        "--out-report",
        default="docs/experiments/audio_labels_v3_finalize_report.md",
        help="Markdown report for finalization cleanup.",
    )
    parser.add_argument("--check-out", default="results/check_audio_labels_v3")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing out-labels file.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].astype(str).value_counts(dropna=False).sort_index().items()
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None_\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in values) + " |")
    return "\n".join(lines) + "\n"


def row_examples(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    columns = [
        column
        for column in ["clip_id", "audio_file", "start_time", "end_time", "label", "split", "hard_negative_source"]
        if column in frame.columns
    ]
    return frame.loc[:, columns].head(limit).to_dict("records")


def resolve_audio_path(audio_file: object) -> Path:
    value = Path(str(audio_file))
    if value.is_absolute():
        return value
    root_relative = ROOT / value
    if root_relative.exists():
        return root_relative
    return value


def audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def split_label_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    if not {"split", "label"}.issubset(frame.columns):
        return {}
    pivot = (
        frame.groupby(["split", "label"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=sorted(ALLOWED_SPLITS), fill_value=0)
    )
    for label in sorted(ALLOWED_LABELS):
        if label not in pivot.columns:
            pivot[label] = 0
    return {
        str(split): {str(label): int(pivot.loc[split, label]) for label in sorted(ALLOWED_LABELS)}
        for split in pivot.index
    }


def assert_safe_paths(candidate: Path, out_labels: Path, base_labels: Path, overwrite: bool) -> None:
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate labels not found: {candidate}")
    if not base_labels.exists():
        raise FileNotFoundError(f"Base v2 labels not found: {base_labels}")
    if candidate.resolve() == out_labels.resolve():
        raise ValueError("--out-labels must not be the candidate input path")
    if base_labels.resolve() == out_labels.resolve():
        raise ValueError("--out-labels must not overwrite audio_labels_v2.csv")
    raw_path = (ROOT / "data" / "raw").resolve()
    try:
        out_labels.resolve().relative_to(raw_path)
    except ValueError:
        pass
    else:
        raise ValueError("--out-labels must not be under data/raw")
    if out_labels.exists() and not overwrite:
        raise FileExistsError(f"Output labels already exist; pass --overwrite to replace: {out_labels}")


def cleanup_metadata(candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(candidate.columns))
    if missing_columns:
        raise ValueError(f"Candidate labels missing required columns: {missing_columns}")

    protected_before = candidate.loc[:, PROTECTED_COLUMNS].copy(deep=True)
    output = candidate.copy(deep=True)
    fixes: dict[str, Any] = {}

    sample_weight_blank = output["sample_weight"].astype(str).str.strip().eq("")
    fixes["sample_weight_filled_1_0"] = {
        "count": int(sample_weight_blank.sum()),
        "examples": row_examples(output[sample_weight_blank]),
    }
    output.loc[sample_weight_blank, "sample_weight"] = "1.0"
    sample_weight_numeric = pd.to_numeric(output["sample_weight"], errors="coerce")
    if sample_weight_numeric.isna().any():
        bad_rows = output[sample_weight_numeric.isna()]
        raise ValueError(f"sample_weight contains non-numeric values after cleanup: {row_examples(bad_rows)}")
    output["sample_weight"] = sample_weight_numeric.astype(float)

    hard_negative = output["hard_negative"].map(parse_bool)
    hard_source_blank = output["hard_negative_source"].astype(str).str.strip().eq("")
    legacy_hard_negative_mask = hard_negative & hard_source_blank
    fixes["hard_negative_source_filled_legacy_hard_negative"] = {
        "count": int(legacy_hard_negative_mask.sum()),
        "examples": row_examples(output[legacy_hard_negative_mask]),
    }
    output.loc[legacy_hard_negative_mask, "hard_negative_source"] = "legacy_hard_negative"

    protected_after = output.loc[:, PROTECTED_COLUMNS]
    changed_protected = [
        column
        for column in PROTECTED_COLUMNS
        if not protected_before[column].astype(str).equals(protected_after[column].astype(str))
    ]
    if changed_protected:
        raise RuntimeError(f"Protected columns changed unexpectedly: {changed_protected}")
    if len(output) != len(candidate):
        raise RuntimeError(f"Row count changed unexpectedly: candidate={len(candidate)} output={len(output)}")

    fixes["protected_columns_unchanged"] = True
    fixes["candidate_rows"] = int(len(candidate))
    fixes["output_rows"] = int(len(output))
    return output, fixes


def run_dataset_check(labels: pd.DataFrame, labels_path: Path, out_dir: Path) -> dict[str, Any]:
    ensure_dir(out_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(labels.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    starts = pd.to_numeric(labels.get("start_time", pd.Series([], dtype=object)), errors="coerce")
    ends = pd.to_numeric(labels.get("end_time", pd.Series([], dtype=object)), errors="coerce")
    if not missing_columns:
        if starts.isna().any() or ends.isna().any():
            errors.append(f"Rows with non-numeric start_time/end_time: {int((starts.isna() | ends.isna()).sum())}")
        if (starts < 0).any():
            errors.append(f"Rows with start_time < 0: {int((starts < 0).sum())}")
        if (ends <= starts).any():
            errors.append(f"Rows with end_time <= start_time: {int((ends <= starts).sum())}")

        invalid_labels = sorted(set(labels["label"].astype(str)) - ALLOWED_LABELS)
        invalid_splits = sorted(set(labels["split"].astype(str)) - ALLOWED_SPLITS)
        if invalid_labels:
            errors.append(f"Invalid label values: {invalid_labels}")
        if invalid_splits:
            errors.append(f"Invalid split values: {invalid_splits}")

    duplicate_clip_rows = int(labels["clip_id"].astype(str).duplicated(keep=False).sum()) if "clip_id" in labels else 0
    if duplicate_clip_rows:
        errors.append(f"Duplicate clip_id rows: {duplicate_clip_rows}")

    duplicate_audio_segment_rows = 0
    if {"audio_file", "start_time", "end_time"}.issubset(labels.columns):
        duplicate_key = pd.DataFrame(
            {
                "audio_file": labels["audio_file"].astype(str),
                "start": starts.round(6),
                "end": ends.round(6),
            }
        )
        duplicate_audio_segment_rows = int(duplicate_key.duplicated(keep=False).sum())
        if duplicate_audio_segment_rows:
            errors.append(f"Duplicate audio_file + start_time + end_time rows: {duplicate_audio_segment_rows}")

    leaking_people: dict[str, list[str]] = {}
    if {"person_id", "split"}.issubset(labels.columns):
        for person_id, group in labels.groupby(labels["person_id"].astype(str), dropna=False):
            splits = sorted(set(group["split"].astype(str)) - {""})
            if len(splits) > 1:
                leaking_people[str(person_id)] = splits
        if leaking_people:
            errors.append(f"person_id appears in multiple splits: {len(leaking_people)} people")

    audio_missing_rows: list[dict[str, Any]] = []
    duration_overrun_rows: list[dict[str, Any]] = []
    duration_cache: dict[str, float | None] = {}
    if "audio_file" in labels.columns:
        for audio_file in sorted(set(labels["audio_file"].astype(str))):
            path = resolve_audio_path(audio_file)
            duration_cache[audio_file] = audio_duration_seconds(path) if path.exists() else None
        for index, row in labels.iterrows():
            audio_file = str(row["audio_file"])
            path = resolve_audio_path(audio_file)
            if not path.exists():
                audio_missing_rows.append(
                    {"row": int(index), "clip_id": row.get("clip_id", ""), "audio_file": audio_file}
                )
                continue
            duration = duration_cache.get(audio_file)
            if duration is not None and not pd.isna(ends.iloc[index]) and float(ends.iloc[index]) > duration + 1e-6:
                duration_overrun_rows.append(
                    {
                        "row": int(index),
                        "clip_id": row.get("clip_id", ""),
                        "audio_file": audio_file,
                        "end_time": float(ends.iloc[index]),
                        "duration": round(float(duration), 6),
                    }
                )
        if audio_missing_rows:
            errors.append(f"Missing audio_file rows: {len(audio_missing_rows)}")
        if duration_overrun_rows:
            errors.append(f"Rows where end_time exceeds WAV duration: {len(duration_overrun_rows)}")

    sample_weight = pd.to_numeric(labels.get("sample_weight", pd.Series([""] * len(labels))), errors="coerce")
    sample_weight_missing = int(sample_weight.isna().sum())
    sample_weight_non_positive = int((sample_weight.notna() & (sample_weight <= 0)).sum())
    if sample_weight_missing:
        errors.append(f"sample_weight missing/non-numeric rows: {sample_weight_missing}")
    if sample_weight_non_positive:
        errors.append(f"sample_weight <= 0 rows: {sample_weight_non_positive}")

    hard_negative = labels.get("hard_negative", pd.Series([""] * len(labels))).map(parse_bool)
    hard_source = labels.get("hard_negative_source", pd.Series([""] * len(labels))).astype(str)
    hard_negative_empty_source = int((hard_negative & hard_source.str.strip().eq("")).sum())
    if hard_negative_empty_source:
        errors.append(f"hard_negative rows with empty hard_negative_source: {hard_negative_empty_source}")

    source_audio = labels.get("source_audio", pd.Series([""] * len(labels))).astype(str)
    source_audio_non_empty = source_audio[source_audio.str.strip().ne("")]
    source_audio_counts = source_audio_non_empty.value_counts()

    report: dict[str, Any] = {
        "status": "passed" if not errors else "failed",
        "labels_csv": str(labels_path),
        "rows": int(len(labels)),
        "label_counts": value_counts(labels, "label"),
        "split_counts": value_counts(labels, "split"),
        "split_label_counts": split_label_counts(labels),
        "hard_negative_count": int(hard_negative.sum()),
        "hard_negative_source_counts": value_counts(labels, "hard_negative_source"),
        "source_audio": {
            "non_empty_rows": int(source_audio_non_empty.shape[0]),
            "unique_non_empty": int(source_audio_non_empty.nunique()),
            "max_samples_per_source_audio": int(source_audio_counts.max()) if not source_audio_counts.empty else 0,
            "top_source_audio_counts": {str(key): int(value) for key, value in source_audio_counts.head(20).items()},
        },
        "sample_weight": {
            "missing_rows": sample_weight_missing,
            "numeric_positive_rows": int((sample_weight > 0).sum()),
            "non_positive_rows": sample_weight_non_positive,
        },
        "hard_negative_source": {
            "empty_on_hard_negative_rows": hard_negative_empty_source,
        },
        "audio_files": {
            "unique": int(labels["audio_file"].astype(str).nunique()) if "audio_file" in labels else 0,
            "missing_rows": int(len(audio_missing_rows)),
            "duration_overrun_rows": int(len(duration_overrun_rows)),
            "missing_examples": audio_missing_rows[:20],
            "duration_overrun_examples": duration_overrun_rows[:20],
        },
        "duplicate_clip_id_rows": duplicate_clip_rows,
        "duplicate_audio_segment_rows": duplicate_audio_segment_rows,
        "person_id_leakage_count": int(len(leaking_people)),
        "person_id_leakage_examples": dict(list(leaking_people.items())[:20]),
        "errors": errors,
        "warnings": warnings,
    }
    write_json(report, out_dir / "report.json")
    write_dataset_check_md(report, out_dir / "report.md")
    return report


def write_dataset_check_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# audio_labels_v3 Dataset Check",
        "",
        f"- status: {report['status']}",
        f"- rows: {report['rows']}",
        f"- labels_csv: `{report['labels_csv']}`",
        "",
        "## Pass Criteria",
        "",
        f"- audio missing: `{report['audio_files']['missing_rows']}`",
        f"- duplicate audio segment: `{report['duplicate_audio_segment_rows']}`",
        f"- person_id split leakage: `{report['person_id_leakage_count']}`",
        f"- sample_weight missing_rows: `{report['sample_weight']['missing_rows']}`",
        f"- sample_weight non_positive_rows: `{report['sample_weight']['non_positive_rows']}`",
        f"- hard_negative empty source rows: `{report['hard_negative_source']['empty_on_hard_negative_rows']}`",
        f"- label_counts: `{report['label_counts']}`",
        f"- split_counts: `{report['split_counts']}`",
        "",
        "## Split x Label",
        "",
    ]
    split_rows = []
    for split, counts in report.get("split_label_counts", {}).items():
        row = {"split": split}
        row.update(counts)
        split_rows.append(row)
    lines.append(markdown_table(split_rows, ["split", "cough", "non_cough"]))
    lines.extend(
        [
            "## Hard Negative",
            "",
            f"- hard_negative_count: `{report['hard_negative_count']}`",
            f"- hard_negative_source_counts: `{report['hard_negative_source_counts']}`",
            "",
            "## Source Audio",
            "",
            f"- unique_non_empty: `{report['source_audio']['unique_non_empty']}`",
            f"- max_samples_per_source_audio: `{report['source_audio']['max_samples_per_source_audio']}`",
            "",
            "## Errors",
            "",
        ]
    )
    lines.extend([f"- {error}" for error in report["errors"]] if report["errors"] else ["- none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in report["warnings"]] if report["warnings"] else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_finalize_report(
    path: Path,
    candidate_path: Path,
    out_labels: Path,
    fixes: dict[str, Any],
    check_report: dict[str, Any],
) -> None:
    ensure_dir(path.parent)
    lines = [
        "# audio_labels_v3 Finalize Report",
        "",
        "## Inputs",
        "",
        f"- candidate: `{candidate_path}`",
        f"- output labels: `{out_labels}`",
        "",
        "## Cleanup Rules Applied",
        "",
        f"- sample_weight filled with 1.0: `{fixes['sample_weight_filled_1_0']['count']}`",
        (
            "- hard_negative_source filled with legacy_hard_negative: "
            f"`{fixes['hard_negative_source_filled_legacy_hard_negative']['count']}`"
        ),
        f"- protected columns unchanged: `{fixes['protected_columns_unchanged']}`",
        f"- candidate rows: `{fixes['candidate_rows']}`",
        f"- output rows: `{fixes['output_rows']}`",
        "",
        "## sample_weight Fix Examples",
        "",
        markdown_table(
            fixes["sample_weight_filled_1_0"]["examples"],
            ["clip_id", "audio_file", "start_time", "end_time", "label", "split", "hard_negative_source"],
        ),
        "## hard_negative_source Fix Examples",
        "",
        markdown_table(
            fixes["hard_negative_source_filled_legacy_hard_negative"]["examples"],
            ["clip_id", "audio_file", "start_time", "end_time", "label", "split", "hard_negative_source"],
        ),
        "## Dataset Check",
        "",
        f"- status: `{check_report['status']}`",
        f"- audio missing: `{check_report['audio_files']['missing_rows']}`",
        f"- duplicate audio segment: `{check_report['duplicate_audio_segment_rows']}`",
        f"- person_id split leakage: `{check_report['person_id_leakage_count']}`",
        f"- sample_weight missing_rows: `{check_report['sample_weight']['missing_rows']}`",
        f"- sample_weight non_positive_rows: `{check_report['sample_weight']['non_positive_rows']}`",
        f"- hard_negative empty source rows: `{check_report['hard_negative_source']['empty_on_hard_negative_rows']}`",
        f"- label_counts: `{check_report['label_counts']}`",
        f"- split_counts: `{check_report['split_counts']}`",
        f"- hard_negative_source_counts: `{check_report['hard_negative_source_counts']}`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {error}" for error in check_report["errors"]] if check_report["errors"] else ["- none"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_path = resolve_path(args.candidate)
    out_labels = resolve_path(args.out_labels)
    base_labels = resolve_path(args.base_labels)
    out_report = resolve_path(args.out_report)
    check_out = resolve_path(args.check_out)

    assert_safe_paths(candidate_path, out_labels, base_labels, overwrite=args.overwrite)

    candidate = read_csv(candidate_path)
    finalized, fixes = cleanup_metadata(candidate)

    ensure_dir(out_labels.parent)
    finalized.to_csv(out_labels, index=False, encoding="utf-8")

    check_report = run_dataset_check(finalized, out_labels, check_out)
    write_finalize_report(out_report, candidate_path, out_labels, fixes, check_report)

    summary = {
        "status": "passed" if check_report["status"] == "passed" else "failed",
        "candidate": str(candidate_path),
        "out_labels": str(out_labels),
        "finalize_report": str(out_report),
        "check_report_md": str(check_out / "report.md"),
        "check_report_json": str(check_out / "report.json"),
        "fixes": fixes,
        "dataset_check": {
            "status": check_report["status"],
            "audio_missing": check_report["audio_files"]["missing_rows"],
            "duplicate_audio_segment_rows": check_report["duplicate_audio_segment_rows"],
            "person_id_leakage_count": check_report["person_id_leakage_count"],
            "sample_weight_missing_rows": check_report["sample_weight"]["missing_rows"],
            "sample_weight_non_positive_rows": check_report["sample_weight"]["non_positive_rows"],
            "hard_negative_empty_source_rows": check_report["hard_negative_source"]["empty_on_hard_negative_rows"],
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if check_report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
