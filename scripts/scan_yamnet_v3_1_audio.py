from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import yamnet_scan_audio as yamnet_scan
from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import iter_window_ranges, load_audio_mono, pad_window, resolve_path


YAMNET_COLUMNS = [
    "original_path",
    "audio_file",
    "metadata_json",
    "source_domain",
    "session_id",
    "person_id",
    "scenario_hint",
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "start_time",
    "end_time",
    "yamnet_top1",
    "yamnet_top1_score",
    "yamnet_top5",
    "yamnet_cough_score",
    "yamnet_sneeze_score",
    "yamnet_speech_score",
    "yamnet_laughter_score",
    "yamnet_breathing_score",
    "yamnet_throat_like_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score v3.1 mining audio with YAMNet.")
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def top_n(scores: np.ndarray, display_names: list[str], n: int = 5) -> list[tuple[str, float]]:
    order = np.argsort(scores)[::-1][:n]
    return [(display_names[int(index)], float(scores[int(index)])) for index in order]


def top5_token(items: list[tuple[str, float]]) -> str:
    return "|".join(f"{label}:{score:.6f}" for label, score in items)


def as_float(row: pd.Series, name: str, fallback: float = 0.0) -> float:
    value = row.get(name, fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def score_audio_file(
    row: pd.Series,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    pcm_defaults: dict[str, Any],
    tf_module,
    yamnet_model,
    display_names: list[str],
    class_groups: dict[str, list[int]],
) -> list[dict[str, object]]:
    audio_ref = str(row.get("audio_file", row.get("original_path", "")))
    audio_path = resolve_path(audio_ref)
    audio = load_audio_mono(audio_path, sample_rate, pcm_defaults)
    window_size = int(round(sample_rate * window_sec))

    has_explicit_window = "window_start_sec" in row.index and "window_end_sec" in row.index
    if has_explicit_window and str(row.get("window_start_sec", "")) != "" and str(row.get("window_end_sec", "")) != "":
        start_time = as_float(row, "window_start_sec")
        end_time = as_float(row, "window_end_sec")
        start_sample = int(round(start_time * sample_rate))
        end_sample = int(round(end_time * sample_rate))
        if end_sample <= start_sample:
            end_sample = start_sample + window_size
        ranges = [(start_sample, end_sample)]
    else:
        ranges = iter_window_ranges(audio.size, sample_rate, window_sec, hop_sec)

    rows: list[dict[str, object]] = []
    for index, (start_sample, end_sample) in enumerate(ranges):
        start_time = start_sample / float(sample_rate)
        end_time = min(end_sample, audio.size) / float(sample_rate)
        chunk = pad_window(audio[start_sample:end_sample], window_size)
        scores = yamnet_scan.yamnet_scores(tf_module, yamnet_model, chunk)
        top5 = top_n(scores, display_names, n=5)
        rows.append(
            {
                "original_path": row.get("original_path", audio_ref),
                "audio_file": audio_ref,
                "metadata_json": row.get("metadata_json", ""),
                "source_domain": row.get("source_domain", ""),
                "session_id": row.get("session_id", ""),
                "person_id": row.get("person_id", ""),
                "scenario_hint": row.get("scenario_hint", ""),
                "window_id": row.get("window_id", f"{audio_path.stem}_{index:06d}"),
                "window_start_sec": round(as_float(row, "window_start_sec", start_time), 6),
                "window_end_sec": round(as_float(row, "window_end_sec", end_time), 6),
                "start_time": round(start_time, 6),
                "end_time": round(end_time, 6),
                "yamnet_top1": top5[0][0],
                "yamnet_top1_score": round(top5[0][1], 6),
                "yamnet_top5": top5_token(top5),
                "yamnet_cough_score": round(yamnet_scan.group_score(scores, class_groups.get("cough", [])), 6),
                "yamnet_sneeze_score": round(yamnet_scan.group_score(scores, class_groups.get("sneeze", [])), 6),
                "yamnet_speech_score": round(yamnet_scan.group_score(scores, class_groups.get("speech", [])), 6),
                "yamnet_laughter_score": round(yamnet_scan.group_score(scores, class_groups.get("laughter", [])), 6),
                "yamnet_breathing_score": round(yamnet_scan.group_score(scores, class_groups.get("breathing", [])), 6),
                "yamnet_throat_like_score": round(
                    yamnet_scan.group_score(scores, class_groups.get("throat_like", [])),
                    6,
                ),
            }
        )
    return rows


def write_summary_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# YAMNet weak scoring summary",
        "",
        f"- manifest: `{report['manifest_csv']}`",
        f"- scored_manifest: `{report['out_csv']}`",
        f"- scanned_sessions: `{report['scanned_sessions']}`",
        f"- scanned_windows: `{report['score_rows']}`",
        f"- error_rows: `{report['error_rows']}`",
        "",
        "## Top YAMNet classes",
        "",
    ]
    for label, count in report.get("top1_distribution", []):
        lines.append(f"- {label}: `{count}`")
    lines.extend([
        "",
        "## Score summary",
        "",
    ])
    for name, stats in report.get("score_summary", {}).items():
        lines.append(
            f"- {name}: min={stats['min']:.6f}, mean={stats['mean']:.6f}, median={stats['median']:.6f}, max={stats['max']:.6f}"
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- weak scoring only",
        "- no manual_decision",
        "- no candidate csv",
        "- no review clips",
        "- no training",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")

    config = load_config(args.config)
    manifest_path = resolve_path(args.manifest or config["paths"]["manifest_csv"])
    out_path = resolve_path(args.out or config["paths"]["yamnet_scores_csv"])
    report_path = resolve_path(args.report or out_path.with_name(f"{out_path.stem}_report.json"))
    summary_path = resolve_path(args.summary_md or out_path.with_name("summary.md"))
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {out_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    rules_config = load_config(resolve_path(config["yamnet"].get("rules_config", "configs/yamnet_hard_mining.yaml")))
    model_handle = str(config["yamnet"].get("model_handle") or rules_config.get("yamnet", {}).get("model_handle"))
    tf_module, yamnet_model, class_map = yamnet_scan.load_yamnet(
        model_handle,
        resolve_path(config["yamnet"].get("tfhub_cache_dir", ".cache/tfhub")),
    )
    display_names = [str(name) for name in class_map["display_name"].tolist()]
    groups_config = rules_config.get("yamnet", {}).get("class_groups", yamnet_scan.DEFAULT_GROUPS)
    class_groups, missing_classes = yamnet_scan.match_class_groups(display_names, groups_config)

    audio_cfg = config["audio"]
    sample_rate = int(audio_cfg.get("sample_rate", 16000))
    window_sec = float(audio_cfg.get("window_seconds", 1.0))
    hop_sec = float(audio_cfg.get("hop_seconds", 0.5))
    pcm_defaults = dict(config.get("mining", {}).get("pcm_defaults", {}))
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        try:
            rows.extend(
                score_audio_file(
                    row,
                    sample_rate=sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    pcm_defaults=pcm_defaults,
                    tf_module=tf_module,
                    yamnet_model=yamnet_model,
                    display_names=display_names,
                    class_groups=class_groups,
                )
            )
        except Exception as exc:
            errors.append({"audio_file": row.get("audio_file", row.get("original_path", "")), "error": str(exc)})

    ensure_dir(out_path.parent)
    scored = pd.DataFrame(rows, columns=YAMNET_COLUMNS)
    scored.to_csv(out_path, index=False)
    pd.DataFrame(errors).to_csv(out_path.with_name(f"{out_path.stem}_errors.csv"), index=False)

    top1_distribution = (
        scored["yamnet_top1"].value_counts().head(20).rename_axis("class").reset_index(name="count").values.tolist()
        if not scored.empty
        else []
    )
    score_summary: dict[str, dict[str, float]] = {}
    for column in [
        "yamnet_cough_score",
        "yamnet_speech_score",
        "yamnet_laughter_score",
        "yamnet_throat_like_score",
    ]:
        if column in scored.columns and not scored.empty:
            series = pd.to_numeric(scored[column], errors="coerce").dropna()
            if not series.empty:
                score_summary[column] = {
                    "min": float(series.min()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "max": float(series.max()),
                }

    report = {
        "manifest_csv": str(manifest_path),
        "out_csv": str(out_path),
        "errors_csv": str(out_path.with_name(f"{out_path.stem}_errors.csv")),
        "summary_md": str(summary_path),
        "model_handle": model_handle,
        "score_rows": int(len(rows)),
        "error_rows": int(len(errors)),
        "scanned_sessions": int(manifest["session_id"].nunique()) if "session_id" in manifest.columns else int(len(manifest)),
        "missing_classes": missing_classes,
        "top1_distribution": top1_distribution,
        "score_summary": score_summary,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(summary_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
