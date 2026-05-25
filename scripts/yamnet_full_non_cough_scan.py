from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import yamnet_scan_audio as scan
from src.common.config import load_config
from src.common.io import ensure_dir


PROCESSED_COLUMNS = [
    "audio_file_abs",
    "audio_file",
    "status",
    "windows",
    "candidate_rows",
    "error_rows",
    "duration_sec",
    "elapsed_sec",
]

ERROR_COLUMNS = ["audio_file", "start_time", "end_time", "error"]

TOP_DECISIONS = [
    "hard_negative_review",
    "baseline_suspicious_review",
    "pseudo_cough_candidate",
    "uncertain_review",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable full YAMNet scan for the external non_cough pool."
    )
    parser.add_argument("--audio-dir", default="D:/cough_model_train/DATA/non_cough")
    parser.add_argument("--out-dir", default="data/mining_candidates/yamnet_non_cough_full_resumable")
    parser.add_argument("--baseline-config", default="configs/audio_formal_50_v2.yaml")
    parser.add_argument("--baseline-model", default="models/audio/selected/audio_baseline_v2_best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--rules-config", default="configs/yamnet_hard_mining.yaml")
    parser.add_argument("--yamnet-model", default=None)
    parser.add_argument("--tfhub-cache-dir", default=".cache/tfhub")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-rms", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Optional limiter for staged runs.")
    parser.add_argument("--summary-every", type=int, default=100, help="Write status summary every N files.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Start only if output files do not already exist. This script never deletes old outputs.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def utcish_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    has_header = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def read_processed_set(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["audio_file_abs"] for row in csv.DictReader(f) if row.get("audio_file_abs")}


def decision_rank(row: dict[str, object], decision: str, threshold: float) -> float:
    baseline = float(row.get("baseline_cough_prob") or 0.0)
    cough = float(row.get("yamnet_cough_score") or 0.0)
    if decision in {"hard_negative_review", "baseline_suspicious_review"}:
        return baseline
    if decision == "pseudo_cough_candidate":
        return baseline + cough
    if decision == "uncertain_review":
        return -abs(baseline - threshold)
    return baseline


def update_top_rows(
    top_rows: dict[str, list[dict[str, object]]],
    row: dict[str, object],
    threshold: float,
    top_k: int,
) -> None:
    decision = str(row.get("decision", ""))
    if decision not in top_rows:
        return
    top_rows[decision].append(dict(row))
    if len(top_rows[decision]) > top_k * 4:
        top_rows[decision].sort(
            key=lambda item: decision_rank(item, decision, threshold),
            reverse=True,
        )
        del top_rows[decision][top_k:]


def load_existing_state(
    candidate_csv: Path,
    processed_csv: Path,
    errors_csv: Path,
    threshold: float,
    top_k: int,
) -> tuple[set[str], dict[str, int], dict[str, list[dict[str, object]]], int, int]:
    processed = read_processed_set(processed_csv)
    counts: dict[str, int] = {}
    top_rows = {decision: [] for decision in TOP_DECISIONS}
    candidate_rows = 0
    if candidate_csv.exists() and candidate_csv.stat().st_size > 0:
        with candidate_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                candidate_rows += 1
                decision = str(row.get("decision", ""))
                counts[decision] = counts.get(decision, 0) + 1
                update_top_rows(top_rows, row, threshold, top_k)
    for decision in TOP_DECISIONS:
        top_rows[decision].sort(
            key=lambda item, name=decision: decision_rank(item, name, threshold),
            reverse=True,
        )
        del top_rows[decision][top_k:]

    error_rows = 0
    if errors_csv.exists() and errors_csv.stat().st_size > 0:
        with errors_csv.open("r", newline="", encoding="utf-8") as f:
            error_rows = sum(1 for _ in csv.DictReader(f))
    return processed, counts, top_rows, candidate_rows, error_rows


def ordered_counts(counts: dict[str, int]) -> dict[str, int]:
    ordered = {decision: int(counts[decision]) for decision in scan.DECISION_ORDER if decision in counts}
    for decision in sorted(set(counts) - set(ordered)):
        ordered[decision] = int(counts[decision])
    return ordered


def write_top_candidates(path: Path, top_rows: dict[str, list[dict[str, object]]]) -> None:
    rows: list[dict[str, object]] = []
    for decision in TOP_DECISIONS:
        for row in top_rows.get(decision, []):
            output = dict(row)
            output["top_bucket"] = decision
            rows.append(output)
    fieldnames = ["top_bucket", *scan.DEFAULT_COLUMNS]
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def write_status_files(
    status: str,
    args: argparse.Namespace,
    out_dir: Path,
    candidate_csv: Path,
    errors_csv: Path,
    processed_csv: Path,
    progress_json: Path,
    codex_md: Path,
    top_csv: Path,
    total_files: int,
    processed_files: int,
    candidate_rows: int,
    error_rows: int,
    counts: dict[str, int],
    missing_classes: list[str],
    top_rows: dict[str, list[dict[str, object]]],
    message: str = "",
) -> dict[str, Any]:
    write_top_candidates(top_csv, top_rows)
    summary = {
        "status": status,
        "updated_at": utcish_now(),
        "message": message,
        "audio_dir": str(resolve_path(args.audio_dir)),
        "out_dir": str(out_dir),
        "candidate_csv": str(candidate_csv),
        "errors_csv": str(errors_csv),
        "processed_files_csv": str(processed_csv),
        "top_candidates_csv": str(top_csv),
        "codex_next_input_md": str(codex_md),
        "total_files": int(total_files),
        "processed_files": int(processed_files),
        "remaining_files": int(max(0, total_files - processed_files)),
        "candidate_rows": int(candidate_rows),
        "error_rows": int(error_rows),
        "decision_counts": ordered_counts(counts),
        "missing_yamnet_class_names": missing_classes,
        "threshold": float(args.threshold),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "next_step_hint": (
            "If status is completed, export review clips from candidate_csv. "
            "If status is interrupted, rerun the same command to resume."
        ),
    }
    progress_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Codex Next Input - YAMNet Full Non-Cough Scan",
        "",
        "Paste this block into the next Codex chat if you need follow-up analysis.",
        "",
        f"- status: {summary['status']}",
        f"- updated_at: {summary['updated_at']}",
        f"- message: {summary['message']}",
        f"- audio_dir: `{summary['audio_dir']}`",
        f"- out_dir: `{summary['out_dir']}`",
        f"- candidate_csv: `{summary['candidate_csv']}`",
        f"- errors_csv: `{summary['errors_csv']}`",
        f"- processed_files_csv: `{summary['processed_files_csv']}`",
        f"- top_candidates_csv: `{summary['top_candidates_csv']}`",
        f"- total_files: {summary['total_files']}",
        f"- processed_files: {summary['processed_files']}",
        f"- remaining_files: {summary['remaining_files']}",
        f"- candidate_rows: {summary['candidate_rows']}",
        f"- error_rows: {summary['error_rows']}",
        "",
        "## Decision Counts",
        "",
    ]
    for decision, count in summary["decision_counts"].items():
        lines.append(f"- {decision}: {count}")
    lines.extend(
        [
            "",
            "## Missing YAMNet Classes",
            "",
            ", ".join(missing_classes) if missing_classes else "_None_",
            "",
            "## Next Step",
            "",
            summary["next_step_hint"],
            "",
        ]
    )
    codex_md.write_text("\n".join(lines), encoding="utf-8")
    return summary


def scan_one_file(
    audio_path: Path,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    threshold: float,
    rules: dict[str, Any],
    tf_module,
    yamnet_model,
    display_names: list[str],
    class_groups: dict[str, list[int]],
    torch_module,
    baseline_model,
    baseline_device,
    baseline_audio_config: dict[str, Any],
    cough_index: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, float, str]:
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    window_size = int(round(sample_rate * window_sec))
    audio_label = scan.display_path(audio_path)

    try:
        audio = scan.load_audio(audio_path, sample_rate)
    except Exception as exc:  # pragma: no cover - depends on local files
        errors.append({"audio_file": audio_label, "start_time": "", "end_time": "", "error": str(exc)})
        return rows, errors, 0, 0.0, "read_error"

    ranges = scan.iter_window_ranges(audio.size, sample_rate, window_sec, hop_sec)
    duration_sec = float(audio.size) / float(sample_rate) if sample_rate else 0.0
    if not ranges:
        rows.append(scan.empty_row(audio_path, 0.0, 0.0, "exclude", "empty_audio"))
        return rows, errors, 0, duration_sec, "ok"

    min_fraction = float(rules.get("min_window_fraction", 0.8))
    min_rms = float(rules.get("silence_min_rms", 0.003))
    for start_sample, end_sample in ranges:
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        duration = max(0.0, end_time - start_time)
        chunk_raw = audio[start_sample:end_sample]
        rms = float(math.sqrt(float(np.mean(np.square(chunk_raw)))) if chunk_raw.size else 0.0)

        if duration < window_sec * min_fraction:
            rows.append(scan.empty_row(audio_path, start_time, end_time, "exclude", "window_too_short"))
            continue
        if rms < min_rms:
            rows.append(scan.empty_row(audio_path, start_time, end_time, "exclude", "rms_too_low"))
            continue

        chunk = scan.pad_window(chunk_raw, window_size)
        try:
            y_scores = scan.yamnet_scores(tf_module, yamnet_model, chunk)
            top3 = scan.top_classes(y_scores, display_names)
            values = {
                "yamnet_cough_score": scan.group_score(y_scores, class_groups.get("cough", [])),
                "yamnet_sneeze_score": scan.group_score(y_scores, class_groups.get("sneeze", [])),
                "yamnet_speech_score": scan.group_score(y_scores, class_groups.get("speech", [])),
                "yamnet_laughter_score": scan.group_score(y_scores, class_groups.get("laughter", [])),
                "yamnet_breathing_score": scan.group_score(y_scores, class_groups.get("breathing", [])),
                "yamnet_throat_like_score": scan.group_score(y_scores, class_groups.get("throat_like", [])),
                "baseline_cough_prob": scan.baseline_prob(
                    torch_module,
                    baseline_model,
                    baseline_device,
                    baseline_audio_config,
                    cough_index,
                    chunk,
                ),
            }
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            errors.append(
                {
                    "audio_file": audio_label,
                    "start_time": round(start_time, 6),
                    "end_time": round(end_time, 6),
                    "error": str(exc),
                }
            )
            rows.append(scan.empty_row(audio_path, start_time, end_time, "exclude", "model_inference_failed"))
            continue

        decision, priority, weight, reason = scan.decide_candidate(
            audio_label,
            values,
            rms=rms,
            duration=duration,
            window_sec=window_sec,
            threshold=threshold,
            rules=rules,
        )
        rows.append(
            {
                "sample_id": scan.stable_sample_id(audio_path, start_time, end_time),
                "audio_file": audio_label,
                "start_time": round(start_time, 6),
                "end_time": round(end_time, 6),
                "duration": round(duration, 6),
                "yamnet_top1": top3[0][0],
                "yamnet_top1_score": round(top3[0][1], 6),
                "yamnet_top2": top3[1][0],
                "yamnet_top2_score": round(top3[1][1], 6),
                "yamnet_top3": top3[2][0],
                "yamnet_top3_score": round(top3[2][1], 6),
                "yamnet_cough_score": round(values["yamnet_cough_score"], 6),
                "yamnet_sneeze_score": round(values["yamnet_sneeze_score"], 6),
                "yamnet_speech_score": round(values["yamnet_speech_score"], 6),
                "yamnet_laughter_score": round(values["yamnet_laughter_score"], 6),
                "yamnet_breathing_score": round(values["yamnet_breathing_score"], 6),
                "yamnet_throat_like_score": round(values["yamnet_throat_like_score"], 6),
                "baseline_cough_prob": round(values["baseline_cough_prob"], 6),
                "decision": decision,
                "review_priority": priority,
                "sample_weight": weight,
                "reason": reason,
            }
        )

    elapsed = time.perf_counter() - started
    return rows, errors, len(ranges), duration_sec, "ok"


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")
    if args.summary_every <= 0:
        raise ValueError("--summary-every must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    audio_dir = resolve_path(args.audio_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    candidate_csv = out_dir / "candidate_pool.csv"
    errors_csv = out_dir / "errors.csv"
    processed_csv = out_dir / "processed_files.csv"
    progress_json = out_dir / "scan_progress.json"
    codex_md = out_dir / "codex_next_input.md"
    top_csv = out_dir / "top_candidates_preview.csv"

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if args.restart:
        existing = [path for path in (candidate_csv, errors_csv, processed_csv) if path.exists()]
        if existing:
            raise FileExistsError(
                "--restart was requested, but output files already exist. "
                f"Use a new --out-dir or manually archive/delete after review: {existing}"
            )

    rules_config = load_config(resolve_path(args.rules_config))
    sample_rate = int(args.sample_rate or rules_config.get("project", {}).get("sample_rate", 16000))
    if sample_rate != 16000:
        raise ValueError("YAMNet requires 16000 Hz audio; keep --sample-rate 16000")
    rules = dict(rules_config["decision_rules"])
    if args.min_rms is not None:
        rules["silence_min_rms"] = float(args.min_rms)

    model_handle = args.yamnet_model or str(rules_config.get("yamnet", {}).get("model_handle"))
    if not model_handle or model_handle == "None":
        raise ValueError("YAMNet model handle is missing")

    wav_files = scan.iter_wavs(audio_dir)
    if args.max_files is not None:
        wav_files = wav_files[: args.max_files]
    total_files = len(wav_files)

    processed, counts, top_rows, candidate_rows, error_rows = load_existing_state(
        candidate_csv,
        processed_csv,
        errors_csv,
        float(args.threshold),
        int(args.top_k),
    )
    pending_files = [path for path in wav_files if str(path.resolve()) not in processed]

    torch_module, baseline_model, baseline_device, baseline_audio_config, cough_index = scan.load_baseline(
        resolve_path(args.baseline_config),
        resolve_path(args.baseline_model),
    )
    if int(baseline_audio_config["sample_rate"]) != sample_rate:
        raise ValueError(
            f"Baseline sample_rate={baseline_audio_config['sample_rate']} differs from YAMNet sample_rate={sample_rate}"
        )

    tf_module, yamnet_model, class_map = scan.load_yamnet(model_handle, resolve_path(args.tfhub_cache_dir))
    display_names = [str(name) for name in class_map["display_name"].tolist()]
    class_groups_cfg = rules_config.get("yamnet", {}).get("class_groups", scan.DEFAULT_GROUPS) or scan.DEFAULT_GROUPS
    class_groups, missing_classes = scan.match_class_groups(display_names, class_groups_cfg)

    write_status_files(
        "running",
        args,
        out_dir,
        candidate_csv,
        errors_csv,
        processed_csv,
        progress_json,
        codex_md,
        top_csv,
        total_files,
        len(processed),
        candidate_rows,
        error_rows,
        counts,
        missing_classes,
        top_rows,
        message="scan started or resumed",
    )

    try:
        iterator = scan.progress(pending_files, desc="Full non_cough scan", unit="file")
        for index, audio_path in enumerate(iterator, start=1):
            started = time.perf_counter()
            rows, errors, windows, duration_sec, status = scan_one_file(
                audio_path,
                sample_rate=sample_rate,
                window_sec=float(args.window_sec),
                hop_sec=float(args.hop_sec),
                threshold=float(args.threshold),
                rules=rules,
                tf_module=tf_module,
                yamnet_model=yamnet_model,
                display_names=display_names,
                class_groups=class_groups,
                torch_module=torch_module,
                baseline_model=baseline_model,
                baseline_device=baseline_device,
                baseline_audio_config=baseline_audio_config,
                cough_index=cough_index,
            )
            append_rows(candidate_csv, scan.DEFAULT_COLUMNS, rows)
            append_rows(errors_csv, ERROR_COLUMNS, errors)

            processed_row = {
                "audio_file_abs": str(audio_path.resolve()),
                "audio_file": scan.display_path(audio_path),
                "status": status,
                "windows": windows,
                "candidate_rows": len(rows),
                "error_rows": len(errors),
                "duration_sec": round(duration_sec, 6),
                "elapsed_sec": round(time.perf_counter() - started, 6),
            }
            append_rows(processed_csv, PROCESSED_COLUMNS, [processed_row])
            processed.add(str(audio_path.resolve()))

            candidate_rows += len(rows)
            error_rows += len(errors)
            for row in rows:
                decision = str(row.get("decision", ""))
                counts[decision] = counts.get(decision, 0) + 1
                update_top_rows(top_rows, row, float(args.threshold), int(args.top_k))

            if index % int(args.summary_every) == 0:
                summary = write_status_files(
                    "running",
                    args,
                    out_dir,
                    candidate_csv,
                    errors_csv,
                    processed_csv,
                    progress_json,
                    codex_md,
                    top_csv,
                    total_files,
                    len(processed),
                    candidate_rows,
                    error_rows,
                    counts,
                    missing_classes,
                    top_rows,
                    message=f"processed {len(processed)}/{total_files} files",
                )
                print(
                    f"PROGRESS processed={summary['processed_files']}/{summary['total_files']} "
                    f"rows={summary['candidate_rows']} errors={summary['error_rows']} "
                    f"counts={summary['decision_counts']}"
                )

    except KeyboardInterrupt:
        summary = write_status_files(
            "interrupted",
            args,
            out_dir,
            candidate_csv,
            errors_csv,
            processed_csv,
            progress_json,
            codex_md,
            top_csv,
            total_files,
            len(processed),
            candidate_rows,
            error_rows,
            counts,
            missing_classes,
            top_rows,
            message="KeyboardInterrupt; rerun the same command to resume.",
        )
        print("COPY_FOR_CODEX_BEGIN")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("COPY_FOR_CODEX_END")
        return 130

    summary = write_status_files(
        "completed",
        args,
        out_dir,
        candidate_csv,
        errors_csv,
        processed_csv,
        progress_json,
        codex_md,
        top_csv,
        total_files,
        len(processed),
        candidate_rows,
        error_rows,
        counts,
        missing_classes,
        top_rows,
        message="full scan completed",
    )
    print("COPY_FOR_CODEX_BEGIN")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("COPY_FOR_CODEX_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
