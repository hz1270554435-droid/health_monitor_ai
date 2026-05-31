#!/usr/bin/env python3
"""
v3.2.1 Source Inventory Builder — Stage 2C fixture smoke
========================================================
Minimal, read-only CLI that scans a source audio directory and generates a
source-inventory CSV compliant with the v3.2.1 audio screening schema.

Usage (fixture dry-run):
  python tools/build_audio_source_inventory.py \
    --source-root docs/audio/fixtures/source_inventory_fixture \
    --source-type labeled_cough_folder \
    --person-id p01 \
    --session-id sess_p01_cough_40cm \
    --default-distance-cm 40 \
    --notes "p01 cough at 40cm — review_required" \
    --output-csv docs/audio/fixtures/v3_2_1_source_inventory_fixture.csv

Stage: 2C — fixture smoke only.
No training, no label publication, no data/raw changes, no firmware changes.
Never copies, moves, deletes, resamples, normalizes, or edits audio files.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# Source-type → default field mapping
# ═══════════════════════════════════════════════════════════════════════

# CLI source_type → schema source_type + default field values
SOURCE_TYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "labeled_cough_folder": {
        "schema_source_type": "labeled_cough_session",
        "label": "cough",
        "label_source": "folder_weak",
        "label_confidence": "silver",
        "final_role": "review_required",   # must pass QA before train
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",       # not checked at inventory stage
    },
    "labeled_non_cough_folder": {
        "schema_source_type": "labeled_mixed_session",
        "label": "non_cough",
        "label_source": "folder_weak",
        "label_confidence": "silver",
        "final_role": "review_required",   # must pass QA before train
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",       # not checked at inventory stage
    },
    "board_live_capture": {
        "schema_source_type": "board_live_capture",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "review_required",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "unlabeled_night_audio": {
        "schema_source_type": "unlabeled_night_audio",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "eval_unlabeled",    # night → eval, not mining
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "unlabeled_chat_audio": {
        "schema_source_type": "unlabeled_chat_audio",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "mining_pool",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "unlabeled_indoor_continuous_audio": {
        "schema_source_type": "unlabeled_indoor_continuous_audio",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "mining_pool",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "unlabeled_mixed_audio": {
        "schema_source_type": "unlabeled_mixed_audio",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "mining_pool",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "unlabeled_demo_env_audio": {
        "schema_source_type": "unlabeled_demo_env_audio",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "mining_pool",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
    "pending_rerecord_source": {
        "schema_source_type": "labeled_cough_session",
        "label": "empty",
        "label_source": "unlabeled",
        "label_confidence": "none",
        "final_role": "pending_rerecord",
        "review_status": "not_reviewed",
        "folder_weak_qa_passed": "",
    },
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".webm"}

# ═══════════════════════════════════════════════════════════════════════
# Special-case overrides
# ═══════════════════════════════════════════════════════════════════════

def apply_special_case_overrides(row: dict[str, str], source_root: Path) -> None:
    """
    Apply policy overrides based on person_id, source_type, distance, and
    session_notes.txt content.

    R18: p01 40cm plosive → must not be in train → final_role=pending_rerecord or exclude
    R19: p02 night → label=empty, label_source=unlabeled, not in train
    """
    person_id = row.get("person_id", "")
    source_type_input = row.get("_source_type_input", "")
    distance_cm = row.get("_distance_cm", "")

    # Check session notes for override keywords
    notes_text = ""
    notes_file = source_root / "session_notes.txt"
    if notes_file.exists():
        try:
            notes_text = notes_file.read_text(encoding="utf-8").lower()
        except Exception:
            pass

    # ── p01 40cm plosive risk (R18) ──
    is_p01_40cm = (
        person_id == "p01"
        and (
            distance_cm == "40"
            or "40cm" in notes_text
            or "plosive" in notes_text
            or "spray" in notes_text
        )
    )
    if is_p01_40cm:
        row["final_role"] = "pending_rerecord"
        row["label"] = "empty"
        row["label_source"] = "unlabeled"
        row["label_confidence"] = "none"
        row["quality_flag"] = "fail_plosive_risk"
        row["split"] = ""
        row["notes"] = (row.get("notes", "") + " p01 40cm plosive risk → pending_rerecord").strip()

    # ── p02 night (R19) ──
    is_p02_night = (
        person_id == "p02"
        and source_type_input == "unlabeled_night_audio"
    )
    if is_p02_night:
        row["final_role"] = "eval_unlabeled"
        row["label"] = "empty"
        row["label_source"] = "unlabeled"
        row["label_confidence"] = "none"
        row["split"] = ""
        row["notes"] = (row.get("notes", "") + " p02 night → eval_unlabeled only (R19)").strip()


# ═══════════════════════════════════════════════════════════════════════
# Main builder
# ═══════════════════════════════════════════════════════════════════════

CSV_FIELDS = [
    # Schema required
    "window_id",
    "event_id",
    "event_cluster_id",
    "session_id",
    "person_id",
    "source_inventory_id",
    "audio_file",
    "start_time",
    "end_time",
    "label",
    "label_source",
    "label_confidence",
    "final_role",
    "source_type",
    "split",
    "review_status",
    # Schema optional (meaningful for inventory)
    "folder_weak_qa_passed",
    "quality_flag",
    "notes",
    # Extra tracking fields
    "source_path",
    "group_id",
    "split_guard_group",
    "duration_s",
    "sample_rate",
    "channels",
    "file_ext",
]


def build_inventory(
    source_root: Path,
    source_type: str,
    person_id: str,
    session_id: str,
    default_distance_cm: str,
    notes: str,
    output_csv: Path,
) -> int:
    """Scan source_root, generate schema-compatible inventory CSV. Returns row count."""

    defaults = SOURCE_TYPE_DEFAULTS.get(source_type)
    if defaults is None:
        print(f"ERROR: Unknown source_type '{source_type}'.", file=sys.stderr)
        print(f"       Known types: {sorted(SOURCE_TYPE_DEFAULTS.keys())}", file=sys.stderr)
        return 0

    # Collect audio files
    audio_files: list[Path] = []
    for ext in AUDIO_EXTENSIONS:
        audio_files.extend(sorted(source_root.rglob(f"*{ext}")))
    audio_files.sort()

    if not audio_files:
        print(f"WARNING: No audio files found in {source_root}")
        return 0

    print(f"Source root:    {source_root}")
    print(f"Source type:    {source_type} → {defaults['schema_source_type']}")
    print(f"Person:         {person_id}")
    print(f"Session:        {session_id}")
    print(f"Audio files:    {len(audio_files)}")

    inventory_counter = 0
    rows: list[dict[str, str]] = []

    for idx, wav_path in enumerate(audio_files, start=1):
        # Read audio metadata (read-only — never modify audio)
        try:
            info = sf.info(str(wav_path))
            duration_s = info.duration
            sample_rate = info.samplerate
            channels = info.channels
        except Exception as exc:
            print(f"  WARNING: Cannot read {wav_path.name}: {exc}")
            continue

        inventory_counter += 1
        inv_id = f"inv_{person_id}_{inventory_counter:03d}"
        clip_stem = wav_path.stem
        file_ext = wav_path.suffix.lower()

        # Generate unique IDs
        window_id = f"{inv_id}_full"
        event_id = inv_id  # each source clip = one event at inventory stage
        event_cluster_id = session_id  # all clips in a session share cluster

        # Relative path for reproducibility
        try:
            audio_file_rel = str(wav_path.relative_to(source_root))
        except ValueError:
            audio_file_rel = str(wav_path)

        row = {
            "window_id": window_id,
            "event_id": event_id,
            "event_cluster_id": event_cluster_id,
            "session_id": session_id,
            "person_id": person_id,
            "source_inventory_id": inv_id,
            "audio_file": audio_file_rel,
            "start_time": "0.0",
            "end_time": f"{duration_s:.4f}",
            "label": defaults["label"],
            "label_source": defaults["label_source"],
            "label_confidence": defaults["label_confidence"],
            "final_role": defaults["final_role"],
            "source_type": defaults["schema_source_type"],
            "split": "",
            "review_status": defaults["review_status"],
            "folder_weak_qa_passed": defaults.get("folder_weak_qa_passed", ""),
            "quality_flag": "",
            "notes": notes,
            # Extra tracking fields
            "source_path": str(wav_path.resolve()),
            "group_id": person_id,
            "split_guard_group": person_id,
            "duration_s": f"{duration_s:.4f}",
            "sample_rate": str(sample_rate),
            "channels": str(channels),
            "file_ext": file_ext,
            # Internal fields (not written to CSV)
            "_source_type_input": source_type,
            "_distance_cm": default_distance_cm,
        }

        # Apply special-case overrides
        apply_special_case_overrides(row, wav_path.parent)

        # Clean up internal fields before writing
        clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
        rows.append(clean_row)

    # Write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows written:   {len(rows)} → {output_csv}")

    # Quick summary
    source_types_used = sorted(set(r["source_type"] for r in rows))
    roles_used = sorted(set(r["final_role"] for r in rows))
    print(f"source_types:   {source_types_used}")
    print(f"final_roles:    {roles_used}")

    return len(rows)


# ═══════════════════════════════════════════════════════════════════════
# Batch mode: scan a fixture root with sub-dirs as source-type groups
# ═══════════════════════════════════════════════════════════════════════

def build_inventory_batch(
    fixture_root: Path,
    output_csv: Path,
) -> int:
    """
    Batch mode: scan fixture_root where each sub-directory name is the source_type.
    Person IDs and session IDs are inferred from sub-sub-directory names.

    Expected layout:
      fixture_root/
        labeled_cough_folder/
          p01_cough_40cm/
            *.wav
            session_notes.txt (optional)
          p02_cough_50cm/
            *.wav
        unlabeled_night_audio/
          p02_night/
            *.wav
        ...
    """
    global_counter = 0  # unique across all sessions
    all_rows: list[dict[str, str]] = []

    for source_type_dir in sorted(fixture_root.iterdir()):
        if not source_type_dir.is_dir():
            continue

        source_type = source_type_dir.name
        if source_type not in SOURCE_TYPE_DEFAULTS:
            print(f"  SKIP: unknown source_type '{source_type}' (directory: {source_type_dir.name})")
            continue

        defaults = SOURCE_TYPE_DEFAULTS[source_type]

        for session_dir in sorted(source_type_dir.iterdir()):
            if not session_dir.is_dir():
                continue

            # Parse person_id, session_id from directory name
            dir_name = session_dir.name
            parts = dir_name.split("_", 1)
            person_id = parts[0] if parts else "unknown"
            session_id = dir_name

            # Detect distance from directory name
            distance_cm = ""
            if "40cm" in dir_name:
                distance_cm = "40"
            elif "50cm" in dir_name:
                distance_cm = "50"

            # Read session notes
            notes = ""
            notes_file = session_dir / "session_notes.txt"
            if notes_file.exists():
                try:
                    notes = notes_file.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

            # Collect audio files
            audio_files = []
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(sorted(session_dir.rglob(f"*{ext}")))
            audio_files.sort()

            if not audio_files:
                continue
            for idx, wav_path in enumerate(audio_files, start=1):
                try:
                    info = sf.info(str(wav_path))
                    duration_s = info.duration
                    sample_rate = info.samplerate
                    channels = info.channels
                except Exception as exc:
                    print(f"  WARNING: Cannot read {wav_path.name}: {exc}")
                    continue

                global_counter += 1
                inv_id = f"inv_{person_id}_{global_counter:03d}"
                clip_stem = wav_path.stem
                file_ext = wav_path.suffix.lower()

                window_id = f"{inv_id}_full"
                event_id = inv_id
                event_cluster_id = session_id

                try:
                    audio_file_rel = str(wav_path.relative_to(fixture_root))
                except ValueError:
                    audio_file_rel = str(wav_path)

                row = {
                    "window_id": window_id,
                    "event_id": event_id,
                    "event_cluster_id": event_cluster_id,
                    "session_id": session_id,
                    "person_id": person_id,
                    "source_inventory_id": inv_id,
                    "audio_file": audio_file_rel,
                    "start_time": "0.0",
                    "end_time": f"{duration_s:.4f}",
                    "label": defaults["label"],
                    "label_source": defaults["label_source"],
                    "label_confidence": defaults["label_confidence"],
                    "final_role": defaults["final_role"],
                    "source_type": defaults["schema_source_type"],
                    "split": "",
                    "review_status": defaults["review_status"],
                    "folder_weak_qa_passed": defaults.get("folder_weak_qa_passed", ""),
                    "quality_flag": "",
                    "notes": notes,
                    "source_path": str(wav_path.resolve()),
                    "group_id": person_id,
                    "split_guard_group": person_id,
                    "duration_s": f"{duration_s:.4f}",
                    "sample_rate": str(sample_rate),
                    "channels": str(channels),
                    "file_ext": file_ext,
                    "_source_type_input": source_type,
                    "_distance_cm": distance_cm,
                }

                apply_special_case_overrides(row, session_dir)
                clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
                all_rows.append(clean_row)

            session_count = len(audio_files)
            print(f"  [{source_type}] {session_dir.name}: {session_count} files")

    # Write combined CSV
    total = global_counter
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal rows: {total} → {output_csv}")

    # Summary
    source_types_used = sorted(set(r["source_type"] for r in all_rows))
    roles_used = sorted(set(r["final_role"] for r in all_rows))
    persons = sorted(set(r["person_id"] for r in all_rows))
    print(f"source_types: {source_types_used}")
    print(f"final_roles:  {roles_used}")
    print(f"persons:      {persons}")

    return total


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v3.2.1 Source Inventory Builder — Stage 2C fixture smoke"
    )

    # Single-source mode
    p.add_argument("--source-root", help="Root directory containing audio files for one source-type.")
    p.add_argument("--source-type", help="Source type label (e.g. labeled_cough_folder).")
    p.add_argument("--person-id", help="Person identifier (e.g. p01).")
    p.add_argument("--session-id", help="Session identifier (e.g. sess_p01_cough_40cm).")
    p.add_argument("--default-distance-cm", default="", help="Recording distance in cm.")
    p.add_argument("--notes", default="", help="Notes to attach to all rows.")

    # Batch mode
    p.add_argument("--batch", action="store_true",
                   help="Batch mode: scan a fixture root where each sub-dir is a source_type, "
                        "each sub-sub-dir is a session.")

    # Output
    p.add_argument("--output-csv", required=True, help="Output CSV path.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.batch:
        source_root = Path(args.source_root)
        if not source_root.exists():
            print(f"ERROR: source-root not found: {source_root}", file=sys.stderr)
            return 1
        n = build_inventory_batch(source_root, Path(args.output_csv))
    else:
        if not args.source_root:
            print("ERROR: --source-root is required (or use --batch mode).", file=sys.stderr)
            return 1
        if not args.source_type:
            print("ERROR: --source-type is required (or use --batch mode).", file=sys.stderr)
            return 1
        if not args.person_id:
            print("ERROR: --person-id is required (or use --batch mode).", file=sys.stderr)
            return 1
        if not args.session_id:
            print("ERROR: --session-id is required (or use --batch mode).", file=sys.stderr)
            return 1

        source_root = Path(args.source_root)
        if not source_root.exists():
            print(f"ERROR: source-root not found: {source_root}", file=sys.stderr)
            return 1

        n = build_inventory(
            source_root=source_root,
            source_type=args.source_type,
            person_id=args.person_id,
            session_id=args.session_id,
            default_distance_cm=args.default_distance_cm,
            notes=args.notes,
            output_csv=Path(args.output_csv),
        )

    if n == 0:
        print("WARNING: No rows generated.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
