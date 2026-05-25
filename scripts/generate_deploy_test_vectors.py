from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.models import build_model
from src.common.config import load_config
from src.common.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIC deployment test vectors.")
    parser.add_argument("--config", default="configs/audio_formal_50_v2.yaml")
    parser.add_argument("--manifest", default="data/processed/audio_features_v2/audio_manifest.csv")
    parser.add_argument("--checkpoint", default="models/audio/selected/audio_baseline_v2_best_model.pt")
    parser.add_argument("--onnx-model", default="models/audio/export/audio_baseline_v2.onnx")
    parser.add_argument("--threshold-json", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-sweep", default=None)
    parser.add_argument("--label-metadata", default=None)
    parser.add_argument("--output-dir", default="deploy/test_vectors/audio")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--selection-profile", choices=("auto", "legacy", "v3_hardneg", "fpfix"), default="auto")
    parser.add_argument("--per-group", type=int, default=2)
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to generate expected PyTorch outputs") from exc
    return torch


def require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required to generate expected ONNX outputs") from exc
    return ort


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def optional_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    return resolve_path(path)


def sample_id(row: pd.Series) -> str:
    if "sample_id" in row and pd.notna(row["sample_id"]):
        return str(row["sample_id"])
    return f"{row['clip_id']}_{int(row['window_index']):04d}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def resolve_feature_paths(manifest: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    output = manifest.copy()
    feature_root = manifest_path.parent
    output["feature_path"] = output["feature_path"].map(
        lambda p: str((feature_root / str(p)).resolve()) if not Path(str(p)).is_absolute() else str(p)
    )
    return output


def load_feature(path: Path) -> np.ndarray:
    with np.load(path) as data:
        feature = data["feature"].astype(np.float32)
    if feature.shape != (40, 101):
        raise ValueError(f"Expected feature shape (40, 101), got {feature.shape}: {path}")
    return feature


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def default_threshold_sweep_path(config: dict[str, Any]) -> Path:
    run_name = str(config.get("run", {}).get("name", ""))
    results_dir = str(config.get("paths", {}).get("results_dir", "results/audio"))
    return ROOT / results_dir / run_name / "threshold_sweep.csv"


def read_threshold(config: dict[str, Any], args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)

    threshold_path = optional_path(args.threshold_json)
    if threshold_path and threshold_path.exists():
        threshold_data = json.loads(threshold_path.read_text(encoding="utf-8"))
        return float(threshold_data["threshold"])

    sweep_path = optional_path(args.threshold_sweep) or default_threshold_sweep_path(config)
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        if "recommended" in sweep.columns:
            recommended = sweep[sweep["recommended"].astype(bool)]
            if not recommended.empty:
                return float(recommended.iloc[0]["threshold"])

    return 0.75


def threshold_decisions(cough_prob: float) -> dict[str, str]:
    decisions: dict[str, str] = {}
    for threshold in (0.50, 0.75, 0.90, 0.95):
        key = f"decision_t{int(threshold * 100):02d}"
        decisions[key] = "cough" if cough_prob >= threshold else "non_cough"
    return decisions


def load_label_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Label metadata not found: {path}")
    labels = pd.read_csv(path)
    if "clip_id" not in labels.columns:
        raise ValueError(f"Label metadata is missing clip_id: {path}")
    columns = [
        "clip_id",
        "hard_negative_source",
        "derived_review_class",
        "review_batch",
        "review_id",
        "review_file",
        "source_audio",
        "source_start_time",
        "source_end_time",
        "baseline_cough_prob",
        "yamnet_top1",
        "yamnet_top1_score",
        "mining_role",
        "source_domain",
        "sample_weight",
        "manual_decision",
        "manual_subtype_norm",
        "review_bucket",
        "v3_cough_prob",
        "notes",
    ]
    present = [column for column in columns if column in labels.columns]
    return labels[present].drop_duplicates("clip_id")


def merge_label_metadata(manifest: pd.DataFrame, label_metadata: pd.DataFrame | None) -> pd.DataFrame:
    if label_metadata is None:
        return manifest
    merged = manifest.merge(label_metadata, on="clip_id", how="left", suffixes=("", "_label_meta"))
    if "notes_label_meta" in merged.columns:
        merged["label_notes"] = merged["notes_label_meta"]
    return merged


def select_legacy_rows(evaluated: pd.DataFrame) -> pd.DataFrame:
    correct = evaluated[evaluated["true_label"] == evaluated["expected_pred_label"]].copy()
    cough_correct = (
        correct[correct["true_label"] == "cough"]
        .sort_values(["cough_prob", "sample_id"], ascending=[False, True])
        .head(5)
    )
    non_cough_correct = (
        correct[correct["true_label"] == "non_cough"]
        .sort_values(["cough_prob", "sample_id"], ascending=[True, True])
        .head(5)
    )
    misclassified = (
        evaluated[evaluated["true_label"] != evaluated["expected_pred_label"]]
        .sort_values(["sample_id"])
        .copy()
    )
    if len(cough_correct) < 5:
        raise ValueError(f"Only found {len(cough_correct)} correct cough samples; need 5")
    if len(non_cough_correct) < 5:
        raise ValueError(f"Only found {len(non_cough_correct)} correct non_cough samples; need 5")
    if len(misclassified) != 4:
        raise ValueError(f"Expected exactly 4 v2 misclassified samples, found {len(misclassified)}")

    cough_correct = cough_correct.assign(vector_group="correct_cough")
    non_cough_correct = non_cough_correct.assign(vector_group="correct_non_cough")
    misclassified = misclassified.assign(vector_group="misclassified")
    return pd.concat([cough_correct, non_cough_correct, misclassified], ignore_index=True)


def metadata_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[column].fillna("").astype(str)


def top_distinct_clip_rows(
    frame: pd.DataFrame,
    *,
    count: int,
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    return (
        frame.sort_values(sort_columns, ascending=ascending)
        .drop_duplicates("clip_id")
        .head(count)
        .copy()
    )


def select_v3_hardneg_rows(evaluated: pd.DataFrame, per_group: int) -> pd.DataFrame:
    if per_group <= 0:
        raise ValueError("--per-group must be greater than zero")

    review_class = metadata_text(evaluated, "derived_review_class")
    hard_source = metadata_text(evaluated, "hard_negative_source")
    label = metadata_text(evaluated, "true_label")
    split = metadata_text(evaluated, "split")

    preferred_eval_split = split.isin(["val", "test"])
    cough_pool = evaluated[label.eq("cough")].copy()
    cough_pool = cough_pool.assign(_preferred_split=preferred_eval_split.loc[cough_pool.index].astype(int))
    cough = top_distinct_clip_rows(
        cough_pool,
        count=per_group,
        sort_columns=["_preferred_split", "cough_prob", "sample_id"],
        ascending=[False, False, True],
    ).assign(vector_group="cough")

    clean_mask = label.eq("non_cough") & (
        review_class.eq("clean_non_cough") | hard_source.eq("clean_non_cough")
    )
    clean_pool = evaluated[clean_mask].copy()
    if len(clean_pool.drop_duplicates("clip_id")) < per_group:
        clean_pool = evaluated[label.eq("non_cough") & ~metadata_text(evaluated, "hard_negative").eq("True")].copy()
        clean_pool = clean_pool.assign(selection_note="fallback_non_hard_negative")
    clean = top_distinct_clip_rows(
        clean_pool,
        count=per_group,
        sort_columns=["cough_prob", "sample_id"],
        ascending=[True, True],
    ).assign(vector_group="clean_non_cough")

    model_false_alarm_pool = evaluated[
        label.eq("non_cough")
        & (
            review_class.eq("model_false_alarm_hard_negative")
            | hard_source.eq("model_false_alarm")
        )
    ].copy()
    model_false_alarm = top_distinct_clip_rows(
        model_false_alarm_pool,
        count=per_group,
        sort_columns=["cough_prob", "sample_id"],
        ascending=[False, True],
    ).assign(vector_group="model_false_alarm_hard_negative")

    human_cough_like_pool = evaluated[
        label.eq("non_cough")
        & (
            review_class.eq("human_cough_like_hard_negative")
            | hard_source.eq("human_cough_like")
        )
    ].copy()
    human_cough_like = top_distinct_clip_rows(
        human_cough_like_pool,
        count=per_group,
        sort_columns=["cough_prob", "sample_id"],
        ascending=[False, True],
    ).assign(vector_group="human_cough_like_hard_negative")

    groups = {
        "cough": cough,
        "clean_non_cough": clean,
        "model_false_alarm_hard_negative": model_false_alarm,
        "human_cough_like_hard_negative": human_cough_like,
    }
    missing = [name for name, rows in groups.items() if len(rows) < per_group]
    if missing:
        counts = {name: int(len(rows)) for name, rows in groups.items()}
        raise ValueError(f"Insufficient v3 vector samples for groups {missing}; counts={counts}")

    return pd.concat(groups.values(), ignore_index=True)


def select_fpfix_rows(evaluated: pd.DataFrame, per_group: int) -> pd.DataFrame:
    if per_group <= 0:
        raise ValueError("--per-group must be greater than zero")
    if "mining_role" not in evaluated.columns:
        raise ValueError("fpfix vector selection requires label metadata with mining_role")

    role_specs = {
        "confirmed_positive": (["cough_prob", "sample_id"], [False, True]),
        "fn_positive": (["cough_prob", "sample_id"], [False, True]),
        "fp_hard_negative": (["cough_prob", "sample_id"], [False, True]),
        "mid_negative": (["cough_prob", "sample_id"], [False, True]),
        "clean_negative": (["cough_prob", "sample_id"], [True, True]),
        "legacy_positive": (["cough_prob", "sample_id"], [False, True]),
        "legacy_hard_negative": (["cough_prob", "sample_id"], [False, True]),
        "legacy_public_non_cough": (["cough_prob", "sample_id"], [True, True]),
    }
    role_series = metadata_text(evaluated, "mining_role")
    groups: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for role, (sort_columns, ascending) in role_specs.items():
        pool = evaluated[role_series.eq(role)].copy()
        selected = top_distinct_clip_rows(
            pool,
            count=per_group,
            sort_columns=sort_columns,
            ascending=ascending,
        ).assign(vector_group=role)
        counts[role] = int(len(selected))
        if not selected.empty:
            groups.append(selected)

    missing = [role for role, count in counts.items() if count < per_group]
    if missing:
        raise ValueError(f"Insufficient fpfix vector samples for roles {missing}; counts={counts}")
    return pd.concat(groups, ignore_index=True)


def select_rows(evaluated: pd.DataFrame, profile: str, per_group: int) -> pd.DataFrame:
    if profile == "legacy":
        return select_legacy_rows(evaluated)
    if profile == "v3_hardneg":
        return select_v3_hardneg_rows(evaluated, per_group)
    if profile == "fpfix":
        return select_fpfix_rows(evaluated, per_group)
    if "mining_role" in evaluated.columns and metadata_text(evaluated, "mining_role").ne("").any():
        return select_fpfix_rows(evaluated, per_group)
    if "derived_review_class" in evaluated.columns or "hard_negative_source" in evaluated.columns:
        return select_v3_hardneg_rows(evaluated, per_group)
    return select_legacy_rows(evaluated)


def c_float_literal(value: float) -> str:
    text = f"{float(value):.8g}"
    if ("." not in text) and ("e" not in text.lower()):
        text = f"{text}.0"
    return f"{text}f"


def c_array(values: np.ndarray, name: str) -> str:
    flat = values.astype(np.float32).reshape(-1)
    lines = [f"static const float {name}[AUDIO_TEST_VECTOR_SIZE] = {{"]
    for start in range(0, len(flat), 8):
        chunk = ", ".join(c_float_literal(float(value)) for value in flat[start : start + 8])
        suffix = "," if start + 8 < len(flat) else ""
        lines.append(f"    {chunk}{suffix}")
    lines.append("};")
    return "\n".join(lines)


def write_header(path: Path, header_vectors: list[dict[str, Any]]) -> None:
    count = len(header_vectors)
    expected0 = ", ".join(c_float_literal(float(item["onnx_logits"][0])) for item in header_vectors)
    expected1 = ", ".join(c_float_literal(float(item["onnx_logits"][1])) for item in header_vectors)
    expected_prob = ", ".join(c_float_literal(float(item["cough_prob"])) for item in header_vectors)
    lines = [
        "#ifndef AUDIO_TEST_VECTORS_H",
        "#define AUDIO_TEST_VECTORS_H",
        "",
        "#include <stdint.h>",
        "",
        "#define AUDIO_TEST_VECTOR_BATCH 1",
        "#define AUDIO_TEST_VECTOR_CHANNELS 1",
        "#define AUDIO_TEST_VECTOR_N_MELS 40",
        "#define AUDIO_TEST_VECTOR_TIME_FRAMES 101",
        "#define AUDIO_TEST_VECTOR_SIZE (AUDIO_TEST_VECTOR_N_MELS * AUDIO_TEST_VECTOR_TIME_FRAMES)",
        f"#define AUDIO_TEST_VECTOR_COUNT {count}",
        "#define AUDIO_TEST_VECTOR_EXPECTED_COUNT AUDIO_TEST_VECTOR_COUNT",
        "",
    ]
    for index, item in enumerate(header_vectors):
        lines.append(f"/* {item['sample_id']} true={item['true_label']} expected={item['expected_pred_label']} cough_prob={item['cough_prob']:.6f} */")
        lines.append(c_array(item["feature"], f"audio_test_vector_{index}"))
        lines.append("")
    lines.append("static const float* const audio_test_vectors[AUDIO_TEST_VECTOR_COUNT] = {")
    for index in range(count):
        suffix = "," if index + 1 < count else ""
        lines.append(f"    audio_test_vector_{index}{suffix}")
    lines.append("};")
    lines.append("")
    lines.append(f"static const float audio_test_vector_expected_output0[AUDIO_TEST_VECTOR_COUNT] = {{ {expected0} }};")
    lines.append(f"static const float audio_test_vector_expected_output1[AUDIO_TEST_VECTOR_COUNT] = {{ {expected1} }};")
    lines.append(f"static const float audio_test_vector_expected_cough_prob[AUDIO_TEST_VECTOR_COUNT] = {{ {expected_prob} }};")
    lines.append("")
    lines.append("#endif /* AUDIO_TEST_VECTORS_H */")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, selected: pd.DataFrame, threshold: float, profile: str) -> None:
    counts = selected["vector_group"].value_counts().to_dict()
    content = f"""# Audio Deploy Test Vectors

Input shape: `[1, 1, 40, 101]`

Each sample directory contains:

- `input_feature_float32.npy`: NumPy float32 Log-Mel tensor with shape `[1, 1, 40, 101]`
- `input_feature_float32.bin`: raw little-endian float32 tensor data in NCHW order
- `expected_output.json`: PyTorch and ONNX logits plus thresholded expected label
- `expected_output_pc.csv`: one-row PC expected-output table
- `metadata.csv`: one-row source metadata table

Selection:

Profile: `{profile}`

Group counts: `{counts}`

Threshold: `{threshold:.2f}`

The C header `audio_test_vectors.h` contains all selected float32 input arrays for board-side fixed-vector inference tests.

Total generated samples: {len(selected)}
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    torch = require_torch()
    ort = require_onnxruntime()
    config = load_config(args.config)

    manifest_path = resolve_path(args.manifest)
    checkpoint_path = resolve_path(args.checkpoint)
    onnx_path = resolve_path(args.onnx_model)
    output_dir = ensure_dir(resolve_path(args.output_dir))
    results_dir = ensure_dir(resolve_path(args.results_dir)) if args.results_dir else None

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    threshold = read_threshold(config, args)
    class_names = [config["labels"]["negative"], config["labels"]["positive"]]
    positive_label = str(config["labels"]["positive"])

    label_metadata = load_label_metadata(optional_path(args.label_metadata))
    manifest = merge_label_metadata(resolve_feature_paths(pd.read_csv(manifest_path), manifest_path), label_metadata)
    if "split" not in manifest.columns:
        raise ValueError(f"Manifest is missing split column: {manifest_path}")
    if args.selection_profile == "legacy":
        eval_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
        if eval_rows.empty:
            raise ValueError("No test rows found in manifest")
    else:
        eval_rows = manifest.reset_index(drop=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_classes = checkpoint.get("class_names", class_names)
    model_name = str(checkpoint.get("model_name", config.get("training", {}).get("model", "small_cnn")))
    model = build_model(model_name, num_classes=len(checkpoint_classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if positive_label not in checkpoint_classes:
        raise ValueError(f"Checkpoint class_names does not include positive label '{positive_label}': {checkpoint_classes}")
    positive_index = int(checkpoint_classes.index(positive_label))

    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name

    evaluated_rows: list[dict[str, Any]] = []
    for _, row in eval_rows.iterrows():
        feature_2d = load_feature(Path(str(row["feature_path"])))
        feature_4d = feature_2d[np.newaxis, np.newaxis, :, :]
        tensor = torch.from_numpy(feature_4d)
        with torch.no_grad():
            pytorch_logits = model(tensor).detach().cpu().numpy()
        onnx_logits = onnx_session.run([onnx_output_name], {onnx_input_name: feature_4d})[0]
        probs = softmax(onnx_logits)
        cough_prob = float(probs[0, positive_index])
        pred_label = positive_label if cough_prob >= threshold else str(config["labels"]["negative"])
        item = row.to_dict()
        decisions = threshold_decisions(cough_prob)
        item.update(
            {
                "sample_id": sample_id(row),
                "true_label": str(row["label"] if "label" in row else row.get("task_label", "")),
                "pytorch_logits": pytorch_logits.reshape(-1).astype(float).tolist(),
                "onnx_logits": onnx_logits.reshape(-1).astype(float).tolist(),
                "cough_prob": cough_prob,
                "threshold": threshold,
                "expected_pred_label": pred_label,
                "feature": feature_4d,
                **decisions,
            }
        )
        evaluated_rows.append(item)

    evaluated = pd.DataFrame(evaluated_rows)
    selected = select_rows(evaluated, args.selection_profile, int(args.per_group))

    manifest_rows = []
    header_items = []
    for index, row in selected.iterrows():
        sample_dir = ensure_dir(output_dir / f"{index:02d}_{row['vector_group']}_{safe_name(row['sample_id'])}")
        feature = np.asarray(row["feature"], dtype=np.float32)
        np.save(sample_dir / "input_feature_float32.npy", feature)
        feature.tofile(sample_dir / "input_feature_float32.bin")
        decisions = threshold_decisions(float(row["cough_prob"]))
        expected = {
            "sample_id": row["sample_id"],
            "audio_file": row.get("audio_file", row.get("audio_path", "")),
            "true_label": row["true_label"],
            "pytorch_logits": row["pytorch_logits"],
            "onnx_logits": row["onnx_logits"],
            "cough_prob": float(row["cough_prob"]),
            "threshold": threshold,
            "expected_pred_label": row["expected_pred_label"],
            "threshold_decisions": decisions,
        }
        write_json(sample_dir / "expected_output.json", expected)
        expected_row = {
            "sample_id": row["sample_id"],
            "out0": float(row["onnx_logits"][0]),
            "out1": float(row["onnx_logits"][1]),
            "cough_prob": float(row["cough_prob"]),
            "threshold": threshold,
            "expected_pred_label": row["expected_pred_label"],
            **decisions,
        }
        pd.DataFrame([expected_row]).to_csv(sample_dir / "expected_output_pc.csv", index=False)
        metadata_row = {
            "sample_id": row["sample_id"],
            "vector_group": row["vector_group"],
            "clip_id": row.get("clip_id", ""),
            "audio_file": row.get("audio_file", row.get("audio_path", "")),
            "true_label": row["true_label"],
            "split": row.get("split", ""),
            "hard_negative": row.get("hard_negative", ""),
            "hard_negative_source": row.get("hard_negative_source", ""),
            "derived_review_class": row.get("derived_review_class", ""),
            "mining_role": row.get("mining_role", ""),
            "source_domain": row.get("source_domain", ""),
            "sample_weight": row.get("sample_weight", ""),
            "manual_decision": row.get("manual_decision", ""),
            "manual_subtype_norm": row.get("manual_subtype_norm", ""),
            "review_bucket": row.get("review_bucket", ""),
            "notes": row.get("notes", row.get("label_notes", "")),
        }
        pd.DataFrame([metadata_row]).to_csv(sample_dir / "metadata.csv", index=False)
        manifest_rows.append(
            {
                "sample_dir": display_path(sample_dir),
                "sample_id": row["sample_id"],
                "vector_group": row["vector_group"],
                "audio_file": row.get("audio_file", row.get("audio_path", "")),
                "true_label": row["true_label"],
                "expected_pred_label": row["expected_pred_label"],
                "cough_prob": float(row["cough_prob"]),
                "mining_role": row.get("mining_role", ""),
                "source_domain": row.get("source_domain", ""),
                "sample_weight": row.get("sample_weight", ""),
                "manual_decision": row.get("manual_decision", ""),
                "manual_subtype_norm": row.get("manual_subtype_norm", ""),
                "review_bucket": row.get("review_bucket", ""),
                **decisions,
            }
        )
        header_items.append(
            {
                "sample_id": row["sample_id"],
                "true_label": row["true_label"],
                "expected_pred_label": row["expected_pred_label"],
                "cough_prob": float(row["cough_prob"]),
                "onnx_logits": row["onnx_logits"],
                "feature": feature.reshape(40, 101),
            }
        )

    manifest_frame = pd.DataFrame(manifest_rows)
    manifest_frame.to_csv(output_dir / "test_vectors_manifest.csv", index=False)
    manifest_frame.to_csv(output_dir / "metadata.csv", index=False)
    manifest_frame[
        [
            "sample_id",
            "vector_group",
            "true_label",
            "expected_pred_label",
            "cough_prob",
            "decision_t50",
            "decision_t75",
            "decision_t90",
            "decision_t95",
        ]
    ].to_csv(output_dir / "expected_output_pc.csv", index=False)
    write_header(output_dir / "audio_test_vectors.h", header_items)
    write_readme(output_dir / "README.md", selected, threshold, args.selection_profile)

    if results_dir is not None:
        manifest_frame.to_csv(results_dir / "metadata.csv", index=False)
        manifest_frame[
            [
                "sample_id",
                "vector_group",
                "true_label",
                "expected_pred_label",
                "cough_prob",
                "decision_t50",
                "decision_t75",
                "decision_t90",
                "decision_t95",
            ]
        ].to_csv(results_dir / "expected_output_pc.csv", index=False)
        write_json(
            results_dir / "generation_report.json",
            {
                "config": str(resolve_path(args.config)),
                "manifest": str(manifest_path),
                "checkpoint": str(checkpoint_path),
                "onnx_model": str(onnx_path),
                "label_metadata": str(optional_path(args.label_metadata)) if optional_path(args.label_metadata) else None,
                "model_name": model_name,
                "class_names": list(checkpoint_classes),
                "positive_index": positive_index,
                "threshold": threshold,
                "selection_profile": args.selection_profile,
                "per_group": int(args.per_group),
                "sample_count": int(len(selected)),
                "vector_output_dir": str(output_dir),
            },
        )

    print(f"OK: wrote {len(selected)} deploy test vectors to {output_dir}")
    print(f"OK: wrote {output_dir / 'audio_test_vectors.h'}")
    print(f"OK: wrote {output_dir / 'README.md'}")
    if results_dir is not None:
        print(f"OK: wrote fixed-vector summary to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
