from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.features import log_mel_feature
from src.audio.models import build_small_cnn
from src.common.config import load_config
from src.common.io import ensure_dir, save_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used in minimal environments
    tqdm = None


DEFAULT_COLUMNS = [
    "sample_id",
    "audio_file",
    "start_time",
    "end_time",
    "duration",
    "yamnet_top1",
    "yamnet_top1_score",
    "yamnet_top2",
    "yamnet_top2_score",
    "yamnet_top3",
    "yamnet_top3_score",
    "yamnet_cough_score",
    "yamnet_sneeze_score",
    "yamnet_speech_score",
    "yamnet_laughter_score",
    "yamnet_breathing_score",
    "yamnet_throat_like_score",
    "baseline_cough_prob",
    "decision",
    "review_priority",
    "sample_weight",
    "reason",
]

DECISION_ORDER = [
    "hard_negative_review",
    "baseline_suspicious_review",
    "pseudo_cough_candidate",
    "uncertain_review",
    "pseudo_non_cough_candidate",
    "exclude",
    "unselected",
]


DEFAULT_GROUPS = {
    "cough": ["Cough"],
    "sneeze": ["Sneeze"],
    "speech": ["Speech", "Conversation", "Narration", "Child speech", "Babbling"],
    "laughter": ["Laughter", "Giggle", "Chuckle"],
    "breathing": ["Breathing", "Snoring", "Wheeze", "Respiratory sounds"],
    "throat_like": ["Throat clearing", "Gargling", "Burp", "Snort"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan WAV files with YAMNet and the selected MIC baseline for hard-sample mining."
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--rules-config", default="configs/yamnet_hard_mining.yaml")
    parser.add_argument("--yamnet-model", default=None)
    parser.add_argument(
        "--tfhub-cache-dir",
        default=".cache/tfhub",
        help="Project-local TF Hub cache. Keeps corrupted global temp caches from breaking scans.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-rms", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Smoke-test limiter.")
    parser.add_argument(
        "--docs-report",
        default="docs/experiments/yamnet_hard_mining_v1.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def iter_wavs(audio_dir: Path) -> list[Path]:
    return sorted(path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".wav")


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for audio loading. Install requirements_yamnet.txt.") from exc
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio.astype(np.float32, copy=False)


def iter_window_ranges(audio_size: int, sample_rate: int, window_sec: float, hop_sec: float) -> list[tuple[int, int]]:
    window_size = int(round(sample_rate * window_sec))
    hop_size = int(round(sample_rate * hop_sec))
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("--window-sec and --hop-sec must be positive")
    if audio_size <= 0:
        return []
    if audio_size < window_size:
        return [(0, audio_size)]
    return [(start, start + window_size) for start in range(0, audio_size - window_size + 1, hop_size)]


def pad_window(chunk: np.ndarray, window_size: int) -> np.ndarray:
    if chunk.size >= window_size:
        return chunk[:window_size].astype(np.float32, copy=False)
    return np.pad(chunk, (0, window_size - chunk.size)).astype(np.float32, copy=False)


def stable_sample_id(audio_path: Path, start_time: float, end_time: float) -> str:
    digest = hashlib.sha1(f"{audio_path.resolve()}:{start_time:.3f}:{end_time:.3f}".encode("utf-8")).hexdigest()[:10]
    return f"{audio_path.stem}_{int(round(start_time * 1000)):08d}_{digest}"


def load_yamnet(model_handle: str, cache_dir: Path):
    ensure_dir(cache_dir)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(cache_dir))
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "tensorflow and tensorflow-hub are required for YAMNet scanning. "
            "Use the isolated .venv_yamnet setup in docs/data_mining/yamnet_env_setup.md."
        ) from exc
    model = hub.load(model_handle)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    class_map = pd.read_csv(class_map_path)
    if "display_name" not in class_map.columns:
        raise ValueError(f"YAMNet class map is missing display_name: {class_map_path}")
    return tf, model, class_map


def load_baseline(config_path: Path, model_path: Path):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to score the selected audio_baseline_v2 model.") from exc

    config = load_config(config_path)
    audio_config = dict(config["audio"])
    sample_rate = int(audio_config["sample_rate"])
    device_name = str(config.get("training", {}).get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu")
    if device_name != "auto":
        device = torch.device(device_name)

    if not model_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", [config["labels"]["negative"], config["labels"]["positive"]])
    cough_label = config["labels"]["positive"]
    if cough_label not in class_names:
        raise ValueError(f"Checkpoint class_names does not include '{cough_label}': {class_names}")
    cough_index = int(class_names.index(cough_label))

    model = build_small_cnn(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return torch, model, device, audio_config, cough_index


def match_class_groups(display_names: list[str], configured_groups: dict[str, list[str]]) -> tuple[dict[str, list[int]], list[str]]:
    groups: dict[str, list[int]] = {}
    missing: list[str] = []
    lowered = [name.lower() for name in display_names]
    for group_name, terms in configured_groups.items():
        matched: set[int] = set()
        for term in terms:
            term_lower = term.lower()
            term_matches = {idx for idx, name in enumerate(lowered) if term_lower in name}
            if not term_matches:
                missing.append(term)
            matched.update(term_matches)
        groups[group_name] = sorted(matched)
    return groups, sorted(set(missing))


def yamnet_scores(tf_module, yamnet_model, chunk: np.ndarray) -> np.ndarray:
    scores, _, _ = yamnet_model(tf_module.convert_to_tensor(chunk, dtype=tf_module.float32))
    values = scores.numpy()
    if values.ndim == 1:
        return values.astype(np.float32, copy=False)
    return values.mean(axis=0).astype(np.float32, copy=False)


def group_score(scores: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.max(scores[indices]))


def top_classes(scores: np.ndarray, display_names: list[str]) -> list[tuple[str, float]]:
    order = np.argsort(scores)[::-1][:3]
    return [(display_names[int(index)], float(scores[int(index)])) for index in order]


def baseline_prob(torch_module, model, device, audio_config: dict[str, Any], cough_index: int, chunk: np.ndarray) -> float:
    feature = log_mel_feature(chunk, audio_config)
    tensor = torch_module.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
    with torch_module.no_grad():
        logits = model(tensor)
        probs = torch_module.softmax(logits, dim=1).detach().cpu().numpy()[0]
    return float(probs[cough_index])


def score_is_high(values: dict[str, float], hard_cfg: dict[str, Any]) -> bool:
    return (
        values["yamnet_sneeze_score"] >= float(hard_cfg["yamnet_sneeze_score_min"])
        or values["yamnet_speech_score"] >= float(hard_cfg["yamnet_speech_score_min"])
        or values["yamnet_laughter_score"] >= float(hard_cfg["yamnet_laughter_score_min"])
        or values["yamnet_breathing_score"] >= float(hard_cfg["yamnet_breathing_score_min"])
        or values["yamnet_throat_like_score"] >= float(hard_cfg["yamnet_throat_like_score_min"])
    )


def is_negative_source(audio_file: str, suspicious_cfg: dict[str, Any]) -> bool:
    source = audio_file.lower()
    keywords = suspicious_cfg.get("negative_source_keywords", [])
    return any(str(keyword).lower() in source for keyword in keywords)


def decide_candidate(
    audio_file: str,
    values: dict[str, float],
    rms: float,
    duration: float,
    window_sec: float,
    threshold: float,
    rules: dict[str, Any],
) -> tuple[str, str, float, str]:
    min_fraction = float(rules.get("min_window_fraction", 0.8))
    min_rms = float(rules.get("silence_min_rms", 0.003))
    if duration < window_sec * min_fraction:
        return "exclude", "", 0.0, "window_too_short"
    if rms < min_rms:
        return "exclude", "", 0.0, "rms_too_low"

    hard_cfg = rules["hard_negative_review"]
    if (
        values["baseline_cough_prob"] >= float(hard_cfg["baseline_cough_prob_min"])
        and values["yamnet_cough_score"] < float(hard_cfg["yamnet_cough_score_max"])
        and score_is_high(values, hard_cfg)
    ):
        return (
            "hard_negative_review",
            str(hard_cfg["review_priority"]),
            float(hard_cfg["sample_weight"]),
            str(hard_cfg["reason"]),
        )

    suspicious_cfg = rules["baseline_suspicious_review"]
    if (
        is_negative_source(audio_file, suspicious_cfg)
        and values["baseline_cough_prob"] >= float(suspicious_cfg["baseline_cough_prob_min"])
        and values["yamnet_cough_score"] <= float(suspicious_cfg["yamnet_cough_score_max"])
    ):
        return (
            "baseline_suspicious_review",
            str(suspicious_cfg["review_priority"]),
            float(suspicious_cfg["sample_weight"]),
            str(suspicious_cfg["reason"]),
        )

    pseudo_cfg = rules["pseudo_cough_candidate"]
    if (
        values["baseline_cough_prob"] >= float(pseudo_cfg["baseline_cough_prob_min"])
        and values["yamnet_cough_score"] >= float(pseudo_cfg["yamnet_cough_score_min"])
    ):
        return (
            "pseudo_cough_candidate",
            str(pseudo_cfg["review_priority"]),
            float(pseudo_cfg["sample_weight"]),
            str(pseudo_cfg["reason"]),
        )

    uncertain_cfg = rules["uncertain_review"]
    near_threshold = (
        float(uncertain_cfg["baseline_cough_prob_min"])
        <= values["baseline_cough_prob"]
        <= float(uncertain_cfg["baseline_cough_prob_max"])
    )
    conflict_high = (
        values["baseline_cough_prob"] >= max(threshold, float(uncertain_cfg["conflict_baseline_high"]))
        and values["yamnet_cough_score"] <= float(uncertain_cfg["conflict_yamnet_cough_low"])
    )
    conflict_low = (
        values["baseline_cough_prob"] <= float(uncertain_cfg["conflict_baseline_low"])
        and values["yamnet_cough_score"] >= float(uncertain_cfg["conflict_yamnet_cough_high"])
    )
    if near_threshold or conflict_high or conflict_low:
        return (
            "uncertain_review",
            str(uncertain_cfg["review_priority"]),
            float(uncertain_cfg["sample_weight"]),
            str(uncertain_cfg["reason"]),
        )

    clean_cfg = rules["pseudo_non_cough_candidate"]
    if (
        values["baseline_cough_prob"] <= float(clean_cfg["baseline_cough_prob_max"])
        and values["yamnet_cough_score"] <= float(clean_cfg["yamnet_cough_score_max"])
        and values["yamnet_sneeze_score"] <= float(clean_cfg["yamnet_sneeze_score_max"])
    ):
        return (
            "pseudo_non_cough_candidate",
            str(clean_cfg["review_priority"]),
            float(clean_cfg["sample_weight"]),
            str(clean_cfg["reason"]),
        )

    return "unselected", "", 0.0, "no_candidate_rule_matched"


def empty_row(audio_path: Path, start_time: float, end_time: float, decision: str, reason: str) -> dict[str, object]:
    duration = max(0.0, end_time - start_time)
    row = {column: "" for column in DEFAULT_COLUMNS}
    row.update(
        {
            "sample_id": stable_sample_id(audio_path, start_time, end_time),
            "audio_file": display_path(audio_path),
            "start_time": round(start_time, 6),
            "end_time": round(end_time, 6),
            "duration": round(duration, 6),
            "yamnet_top1_score": 0.0,
            "yamnet_top2_score": 0.0,
            "yamnet_top3_score": 0.0,
            "yamnet_cough_score": 0.0,
            "yamnet_sneeze_score": 0.0,
            "yamnet_speech_score": 0.0,
            "yamnet_laughter_score": 0.0,
            "yamnet_breathing_score": 0.0,
            "yamnet_throat_like_score": 0.0,
            "baseline_cough_prob": 0.0,
            "decision": decision,
            "review_priority": "",
            "sample_weight": 0.0,
            "reason": reason,
        }
    )
    return row


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._\n"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def top_rows(df: pd.DataFrame, decision: str, threshold: float) -> list[dict[str, Any]]:
    subset = df[df["decision"] == decision].copy()
    if subset.empty:
        return []
    if decision in {"hard_negative_review", "baseline_suspicious_review"}:
        subset = subset.sort_values("baseline_cough_prob", ascending=False)
    elif decision == "pseudo_cough_candidate":
        subset["rank_score"] = subset["baseline_cough_prob"] + subset["yamnet_cough_score"]
        subset = subset.sort_values("rank_score", ascending=False)
    elif decision == "uncertain_review":
        subset["rank_score"] = (subset["baseline_cough_prob"] - threshold).abs()
        subset = subset.sort_values("rank_score", ascending=True)
    return subset.head(20)[
        [
            "sample_id",
            "audio_file",
            "start_time",
            "end_time",
            "yamnet_top1",
            "yamnet_top1_score",
            "yamnet_cough_score",
            "baseline_cough_prob",
            "reason",
        ]
    ].round(6).to_dict(orient="records")


def write_markdown_report(report_path: Path, report: dict[str, Any]) -> None:
    ensure_dir(report_path.parent)
    lines = [
        "# YAMNet hard mining v1",
        "",
        "YAMNet is used only for pre-screening. This run does not modify formal labels and does not train a model.",
        "",
        "## Summary",
        "",
        f"- Audio files scanned: {report['audio_files_scanned']}",
        f"- Total windows: {report['total_windows']}",
        f"- Candidate CSV: `{report['candidate_pool_csv']}`",
        f"- Errors CSV: `{report['errors_csv']}`",
        "",
        "## Decision counts",
        "",
    ]
    for decision, count in report["decision_counts"].items():
        lines.append(f"- {decision}: {count}")
    lines.extend(
        [
            "",
            "## Missing YAMNet class names",
            "",
            ", ".join(report["missing_yamnet_class_names"]) if report["missing_yamnet_class_names"] else "_None._",
            "",
            "## Hard negative review top 20",
            "",
            markdown_table(
                report["top_hard_negative_review"],
                [
                    "sample_id",
                    "audio_file",
                    "start_time",
                    "end_time",
                    "yamnet_top1",
                    "yamnet_top1_score",
                    "yamnet_cough_score",
                    "baseline_cough_prob",
                    "reason",
                ],
            ),
            "",
            "## Baseline suspicious review top 20",
            "",
            markdown_table(
                report["top_baseline_suspicious_review"],
                [
                    "sample_id",
                    "audio_file",
                    "start_time",
                    "end_time",
                    "yamnet_top1",
                    "yamnet_top1_score",
                    "yamnet_cough_score",
                    "baseline_cough_prob",
                    "reason",
                ],
            ),
            "",
            "## Pseudo cough candidate top 20",
            "",
            markdown_table(
                report["top_pseudo_cough_candidate"],
                [
                    "sample_id",
                    "audio_file",
                    "start_time",
                    "end_time",
                    "yamnet_top1",
                    "yamnet_top1_score",
                    "yamnet_cough_score",
                    "baseline_cough_prob",
                    "reason",
                ],
            ),
            "",
            "## Uncertain review top 20",
            "",
            markdown_table(
                report["top_uncertain_review"],
                [
                    "sample_id",
                    "audio_file",
                    "start_time",
                    "end_time",
                    "yamnet_top1",
                    "yamnet_top1_score",
                    "yamnet_cough_score",
                    "baseline_cough_prob",
                    "reason",
                ],
            ),
            "",
            "## Review recommendation",
            "",
            "Review `hard_negative_review` first, then `baseline_suspicious_review`, "
            "then `pseudo_cough_candidate`, then `uncertain_review`. "
            "`pseudo_non_cough_candidate` is low priority and should not be promoted without spot checks.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0.0 and 1.0")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when provided")

    audio_dir = resolve_path(args.audio_dir)
    out_dir = ensure_dir(resolve_path(args.out_dir))
    candidate_csv = out_dir / "candidate_pool.csv"
    errors_csv = out_dir / "errors.csv"
    report_json = out_dir / "yamnet_mining_report.json"
    docs_report = resolve_path(args.docs_report)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

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

    torch_module, baseline_model, baseline_device, baseline_audio_config, cough_index = load_baseline(
        resolve_path(args.baseline_config),
        resolve_path(args.baseline_model),
    )
    if int(baseline_audio_config["sample_rate"]) != sample_rate:
        raise ValueError(
            f"Baseline sample_rate={baseline_audio_config['sample_rate']} differs from YAMNet sample_rate={sample_rate}"
        )

    tf_module, yamnet_model, class_map = load_yamnet(model_handle, resolve_path(args.tfhub_cache_dir))
    display_names = [str(name) for name in class_map["display_name"].tolist()]
    class_groups_cfg = rules_config.get("yamnet", {}).get("class_groups", DEFAULT_GROUPS) or DEFAULT_GROUPS
    class_groups, missing_classes = match_class_groups(display_names, class_groups_cfg)

    wav_files = iter_wavs(audio_dir)
    if args.max_files is not None:
        wav_files = wav_files[: args.max_files]

    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    window_size = int(round(sample_rate * args.window_sec))

    for audio_path in progress(wav_files, desc="Scanning WAV", unit="file"):
        try:
            audio = load_audio(audio_path, sample_rate)
        except Exception as exc:  # pragma: no cover - depends on local files
            errors.append({"audio_file": display_path(audio_path), "error": str(exc)})
            continue

        ranges = iter_window_ranges(audio.size, sample_rate, args.window_sec, args.hop_sec)
        if not ranges:
            rows.append(empty_row(audio_path, 0.0, 0.0, "exclude", "empty_audio"))
            continue

        for start_sample, end_sample in ranges:
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            duration = max(0.0, end_time - start_time)
            chunk_raw = audio[start_sample:end_sample]
            rms = float(math.sqrt(float(np.mean(np.square(chunk_raw)))) if chunk_raw.size else 0.0)

            min_fraction = float(rules.get("min_window_fraction", 0.8))
            min_rms = float(rules.get("silence_min_rms", 0.003))
            if duration < args.window_sec * min_fraction:
                rows.append(empty_row(audio_path, start_time, end_time, "exclude", "window_too_short"))
                continue
            if rms < min_rms:
                rows.append(empty_row(audio_path, start_time, end_time, "exclude", "rms_too_low"))
                continue

            chunk = pad_window(chunk_raw, window_size)
            try:
                y_scores = yamnet_scores(tf_module, yamnet_model, chunk)
                top3 = top_classes(y_scores, display_names)
                values = {
                    "yamnet_cough_score": group_score(y_scores, class_groups.get("cough", [])),
                    "yamnet_sneeze_score": group_score(y_scores, class_groups.get("sneeze", [])),
                    "yamnet_speech_score": group_score(y_scores, class_groups.get("speech", [])),
                    "yamnet_laughter_score": group_score(y_scores, class_groups.get("laughter", [])),
                    "yamnet_breathing_score": group_score(y_scores, class_groups.get("breathing", [])),
                    "yamnet_throat_like_score": group_score(y_scores, class_groups.get("throat_like", [])),
                    "baseline_cough_prob": baseline_prob(
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
                        "audio_file": display_path(audio_path),
                        "start_time": round(start_time, 6),
                        "end_time": round(end_time, 6),
                        "error": str(exc),
                    }
                )
                rows.append(empty_row(audio_path, start_time, end_time, "exclude", "model_inference_failed"))
                continue

            decision, priority, weight, reason = decide_candidate(
                display_path(audio_path),
                values,
                rms=rms,
                duration=duration,
                window_sec=float(args.window_sec),
                threshold=float(args.threshold),
                rules=rules,
            )
            row = {
                "sample_id": stable_sample_id(audio_path, start_time, end_time),
                "audio_file": display_path(audio_path),
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
            rows.append(row)

    candidate_df = pd.DataFrame(rows, columns=DEFAULT_COLUMNS)
    candidate_df.to_csv(candidate_csv, index=False)
    errors_df = pd.DataFrame(errors, columns=["audio_file", "start_time", "end_time", "error"])
    errors_df.to_csv(errors_csv, index=False)

    if candidate_df.empty:
        decision_counts: dict[str, int] = {}
    else:
        raw_counts = {str(k): int(v) for k, v in candidate_df["decision"].value_counts().items()}
        decision_counts = {
            decision: raw_counts[decision]
            for decision in DECISION_ORDER
            if decision in raw_counts
        }
        for decision in sorted(set(raw_counts) - set(decision_counts)):
            decision_counts[decision] = raw_counts[decision]

    report = {
        "audio_dir": str(audio_dir),
        "audio_files_scanned": int(len(wav_files)),
        "total_windows": int(len(candidate_df)),
        "candidate_pool_csv": display_path(candidate_csv),
        "errors_csv": display_path(errors_csv),
        "yamnet_mining_report_json": display_path(report_json),
        "docs_report": display_path(docs_report),
        "threshold": float(args.threshold),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "decision_counts": decision_counts,
        "missing_yamnet_class_names": missing_classes,
        "top_hard_negative_review": top_rows(candidate_df, "hard_negative_review", float(args.threshold)),
        "top_baseline_suspicious_review": top_rows(
            candidate_df,
            "baseline_suspicious_review",
            float(args.threshold),
        ),
        "top_pseudo_cough_candidate": top_rows(candidate_df, "pseudo_cough_candidate", float(args.threshold)),
        "top_uncertain_review": top_rows(candidate_df, "uncertain_review", float(args.threshold)),
        "errors": errors[:20],
    }
    save_json(report, report_json)
    write_markdown_report(docs_report, report)
    print(
        f"OK: scanned {len(wav_files)} files, wrote {len(candidate_df)} windows to {candidate_csv}; "
        f"errors={len(errors)}"
    )
    print(json.dumps({"decision_counts": decision_counts, "missing_classes": missing_classes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
