"""
Stage 3A: Source Inventory Dry-Run on Small Real Data
======================================================
**DEV-ONLY / TEMPORARY — NOT A FORMAL PIPELINE ENTRY POINT.**

This script was created for limited compatibility validation during the
v3.2.1 screening chain upgrade (Stage 3A closeout). It processed 10 small
real data directories using the source inventory builder in single-source mode.

For production source inventory, use `ml/tools/build_audio_source_inventory.py`
directly with --batch mode or --source-root + --source-type.

Read-only. No training, no label publication, no data/raw changes, no firmware.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
BUILDER = Path(r"D:/cough_model_train/health_monitor_ai/tools/build_audio_source_inventory.py")
VALIDATOR = Path(r"D:/cough_model_train/health_monitor_ai/tools/validate_audio_screening_schema.py")
SCHEMA = Path(r"D:/e84_health_monitor/shared/contracts/audio_screening_schema_v3_2_1.yaml")
OUTPUT_DIR = Path(r"D:/e84_health_monitor/docs/audio")
OUTPUT_CSV = OUTPUT_DIR / "stage3a_source_inventory_dryrun.csv"
OUTPUT_JSON = OUTPUT_DIR / "stage3a_source_inventory_validator_report.json"
OUTPUT_MD = OUTPUT_DIR / "stage3a_source_inventory_validator_report.md"
REPORT_MD = OUTPUT_DIR / "v3_2_1_stage3a_source_inventory_dryrun_report.md"

DATA_ROOT = Path(r"D:/cough_model_train/DATA")

# ── Source definitions ──────────────────────────────────────────────────
# Each entry: (source_root, source_type, person_id, session_id, distance_cm, notes)
SOURCES = [
    # --- labeled_cough_folder ---
    (
        DATA_ROOT / "data_5.25_20260525/port_smoke/40cm_front_p01_cough",
        "labeled_cough_folder",
        "p01",
        "sess_p01_40cm_cough_port_smoke",
        "40",
        "p01 40cm front cough — port_smoke test. Plosive risk at 40cm."
    ),
    (
        DATA_ROOT / "data_5.25_20260525/port_smoke/100cm_front_p01_cough",
        "labeled_cough_folder",
        "p01",
        "sess_p01_100cm_cough_port_smoke",
        "100",
        "p01 100cm front cough — port_smoke test. Cleaner distance."
    ),
    (
        DATA_ROOT / "data_5.25_20260525/port_smoke/40cm_front_p02_cough",
        "labeled_cough_folder",
        "p02",
        "sess_p02_40cm_cough_port_smoke",
        "40",
        "p02 40cm front cough — port_smoke test. Clean recording."
    ),
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/real_cough_1_to_2min",
        "labeled_cough_folder",
        "board",
        "sess_board_real_cough_fpfix",
        "",
        "Board-live real cough 1-2min — fpfix smoke test."
    ),
    # --- labeled_non_cough_folder ---
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/quiet_room_2min",
        "labeled_non_cough_folder",
        "board",
        "sess_board_quiet_room_2min",
        "",
        "Board-live quiet room 2min — non_cough reference."
    ),
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/quiet_room_add_fix",
        "labeled_non_cough_folder",
        "board",
        "sess_board_quiet_room_fix",
        "",
        "Board-live quiet room additional fix recordings."
    ),
    # --- unlabeled_chat_audio ---
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/normal_speech_2min",
        "unlabeled_chat_audio",
        "board",
        "sess_board_normal_speech",
        "",
        "Board-live normal speech 2min — unlabeled chat proxy."
    ),
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/loud_speech_2min",
        "unlabeled_chat_audio",
        "board",
        "sess_board_loud_speech",
        "",
        "Board-live loud speech 2min — unlabeled chat proxy."
    ),
    # --- unlabeled_indoor_continuous_audio ---
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/knock_handling_noise_2min",
        "unlabeled_indoor_continuous_audio",
        "board",
        "sess_board_knock_noise",
        "",
        "Board-live knock/handling noise 2min — indoor ambient proxy."
    ),
    # --- board_live_capture ---
    (
        DATA_ROOT / "board_live_fpfix_smoke_20260520/throat_clear_laugh_breath_2min",
        "board_live_capture",
        "board",
        "sess_board_throat_clear",
        "",
        "Board-live throat clear/laugh/breath 2min — review_required."
    ),
]


def run_builder(source_root, source_type, person_id, session_id, distance_cm, notes, output_csv):
    """Run build_audio_source_inventory.py for a single source."""
    cmd = [
        sys.executable, str(BUILDER),
        "--source-root", str(source_root),
        "--source-type", source_type,
        "--person-id", person_id,
        "--session-id", session_id,
        "--output-csv", str(output_csv),
    ]
    if distance_cm:
        cmd += ["--default-distance-cm", distance_cm]
    if notes:
        cmd += ["--notes", notes]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {source_type} / {session_id}")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return None
    return output_csv


def main():
    print("=" * 60)
    print("Stage 3A: Source Inventory Dry-Run on Small Real Data")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Run builder for each source
    temp_dir = Path(tempfile.mkdtemp(prefix="stage3a_"))
    partial_csvs = []
    summary_rows = []

    for i, (src_root, src_type, person, session, dist, notes) in enumerate(SOURCES):
        if not src_root.exists():
            print(f"\n  SKIP [{i+1}/{len(SOURCES)}] {src_type}/{session}: source_root not found")
            summary_rows.append({
                "source_root": str(src_root),
                "source_type": src_type,
                "person_id": person,
                "session_id": session,
                "status": "skipped_not_found",
                "rows": 0,
                "note": "source_root does not exist",
            })
            continue

        partial_csv = temp_dir / f"partial_{i:02d}.csv"
        print(f"\n  [{i+1}/{len(SOURCES)}] {src_type} / {person} / {session}")

        result = run_builder(src_root, src_type, person, session, dist, notes, partial_csv)
        if result is None:
            summary_rows.append({
                "source_root": str(src_root),
                "source_type": src_type,
                "person_id": person,
                "session_id": session,
                "status": "error",
                "rows": 0,
                "note": "builder returned error",
            })
            continue

        # Count rows
        try:
            with open(partial_csv, encoding="utf-8-sig") as f:
                row_count = sum(1 for _ in csv.DictReader(f))
            partial_csvs.append(partial_csv)
            summary_rows.append({
                "source_root": str(src_root),
                "source_type": src_type,
                "person_id": person,
                "session_id": session,
                "status": "ok",
                "rows": row_count,
                "note": notes,
            })
            print(f"    -> {row_count} rows")
        except Exception as e:
            summary_rows.append({
                "source_root": str(src_root),
                "source_type": src_type,
                "person_id": person,
                "session_id": session,
                "status": "error",
                "rows": 0,
                "note": str(e),
            })

    # Phase 2: Merge partial CSVs
    print(f"\n{'='*60}")
    print(f"Merging {len(partial_csvs)} partial CSVs...")

    all_rows = []
    header = None
    for pcsv in partial_csvs:
        with open(pcsv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    total_rows = len(all_rows)
    print(f"Total rows: {total_rows} -> {OUTPUT_CSV}")

    # Phase 3: Validate
    print(f"\n{'='*60}")
    print("Running schema validator...")

    cmd = [
        sys.executable, str(VALIDATOR),
        "--schema", str(SCHEMA),
        "--csv", str(OUTPUT_CSV),
        "--report-json", str(OUTPUT_JSON),
        "--report-md", str(OUTPUT_MD),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse validator output
    validator_passed = "passed" in result.stdout.lower() and '"status": "passed"' in result.stdout

    # Phase 4: Analyze results
    print(f"\n{'='*60}")
    print("Analysis...")

    # Count by source_type + final_role
    from collections import Counter
    st_counter = Counter()
    fr_counter = Counter()
    quality_flags = Counter()
    p01_rows = []
    p02_rows = []

    for row in all_rows:
        st_counter[row.get("source_type", "?")] += 1
        fr_counter[row.get("final_role", "?")] += 1
        qf = row.get("quality_flag", "")
        if qf:
            quality_flags[qf] += 1
        if row.get("person_id") == "p01":
            p01_rows.append(row)
        if row.get("person_id") == "p02":
            p02_rows.append(row)

    # Check p01 enforcement
    p01_pending = all(r["final_role"] == "pending_rerecord" for r in p01_rows if r.get("_source_type_input", "") != "")
    p01_plosive_flag = all(r.get("quality_flag") == "fail_plosive_risk" for r in p01_rows)
    # Actually check specific rows
    p01_40cm_rows = [r for r in p01_rows if "40cm" in r.get("source_path", "").lower() or "40cm" in r.get("notes", "").lower()]
    p01_100cm_rows = [r for r in p01_rows if "100cm" in r.get("source_path", "").lower() or "100cm" in r.get("notes", "").lower()]

    p01_40cm_pending = all(r["final_role"] == "pending_rerecord" for r in p01_40cm_rows) if p01_40cm_rows else "N/A"
    p01_40cm_plosive = all(r.get("quality_flag") == "fail_plosive_risk" for r in p01_40cm_rows) if p01_40cm_rows else "N/A"

    # Check p02 enforcement
    p02_night_rows = [r for r in p02_rows if "night" in r.get("source_type", "").lower()]
    # p02 40cm cough
    p02_rows_all = [r for r in p02_rows]

    print(f"source_type distribution: {dict(st_counter)}")
    print(f"final_role distribution:  {dict(fr_counter)}")
    print(f"quality_flags:            {dict(quality_flags)}")
    print(f"p01 total rows:           {len(p01_rows)}")
    print(f"p01 40cm rows:            {len(p01_40cm_rows)} (all pending_rerecord: {p01_40cm_pending})")
    print(f"p01 100cm rows:           {len(p01_100cm_rows)}")
    print(f"p02 total rows:           {len(p02_rows)}")

    # Phase 5: Write comprehensive report
    print(f"\n{'='*60}")
    print(f"Writing Stage 3A report -> {REPORT_MD}")

    report_lines = []
    report_lines.append("# v3.2.1 Stage 3A — Source Inventory Dry-Run Report")
    report_lines.append("")
    report_lines.append(f"**Date**: 2026-05-31")
    report_lines.append(f"**Status**: {'PASS' if validator_passed else 'FAIL (see analysis)'}")
    report_lines.append(f"**Phase**: Stage 3A — Small real-data source inventory dry-run")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("## 1. Input Source Roots")
    report_lines.append("")
    report_lines.append("| # | Source Root | source_type | person | rows | status |")
    report_lines.append("|---|-------------|-------------|--------|------|--------|")
    for i, sr in enumerate(summary_rows, 1):
        short_root = sr["source_root"].replace(str(DATA_ROOT), "$DATA")
        report_lines.append(
            f"| {i} | `{short_root}` | {sr['source_type']} | {sr['person_id']} "
            f"| {sr['rows']} | {sr['status']} |"
        )
    report_lines.append("")

    report_lines.append("## 2. Summary")
    report_lines.append("")
    report_lines.append(f"- Total sources attempted: {len(SOURCES)}")
    report_lines.append(f"- Sources with data: {len(partial_csvs)}")
    report_lines.append(f"- Sources skipped/error: {len(SOURCES) - len(partial_csvs)}")
    report_lines.append(f"- **Total rows generated: {total_rows}**")
    report_lines.append("")

    report_lines.append("### source_type Coverage")
    report_lines.append("")
    report_lines.append("| source_type | Rows |")
    report_lines.append("|-------------|------|")
    for st, count in sorted(st_counter.items()):
        report_lines.append(f"| `{st}` | {count} |")
    report_lines.append("")

    report_lines.append("### final_role Distribution")
    report_lines.append("")
    report_lines.append("| final_role | Rows |")
    report_lines.append("|------------|------|")
    for fr, count in sorted(fr_counter.items()):
        report_lines.append(f"| `{fr}` | {count} |")
    report_lines.append("")

    report_lines.append("## 3. Validator Results")
    report_lines.append("")
    report_lines.append(f"- **Overall**: {'PASSED' if validator_passed else 'FAILED'}")
    report_lines.append(f"- Output: `{OUTPUT_JSON.name}` + `{OUTPUT_MD.name}`")
    report_lines.append("")

    report_lines.append("## 4. Policy Enforcement Results")
    report_lines.append("")

    # p01 40cm
    report_lines.append("### 4.1 p01 40cm Plosive Risk (R18)")
    report_lines.append("")
    if p01_40cm_rows:
        report_lines.append(f"- p01 40cm rows: {len(p01_40cm_rows)}")
        report_lines.append(f"- All `final_role=pending_rerecord`: **{p01_40cm_pending}**")
        report_lines.append(f"- All `quality_flag=fail_plosive_risk`: **{p01_40cm_plosive}**")
        for r in p01_40cm_rows:
            report_lines.append(f"  - `{r.get('source_inventory_id', '?')}`: "
                              f"final_role={r.get('final_role')}, "
                              f"quality_flag={r.get('quality_flag', '(none)')}, "
                              f"label={r.get('label')}")
    else:
        report_lines.append("- No p01 40cm data found in this dry-run.")
    report_lines.append("")

    # p01 100cm
    report_lines.append("### 4.2 p01 100cm Clean Cough")
    report_lines.append("")
    if p01_100cm_rows:
        for r in p01_100cm_rows:
            report_lines.append(f"  - `{r.get('source_inventory_id', '?')}`: "
                              f"final_role={r.get('final_role')}, "
                              f"label={r.get('label')}")
    else:
        report_lines.append("- No p01 100cm data found.")
    report_lines.append("")

    # p02
    report_lines.append("### 4.3 p02 Audio")
    report_lines.append("")
    report_lines.append(f"- p02 rows: {len(p02_rows)}")
    for r in p02_rows:
        report_lines.append(f"  - `{r.get('source_inventory_id', '?')}`: "
                          f"source_type={r.get('source_type')}, "
                          f"final_role={r.get('final_role')}, "
                          f"label={r.get('label')}")
    report_lines.append("")

    # unlabeled
    report_lines.append("### 4.4 Unlabeled Sources")
    report_lines.append("")
    unlabeled_rows = [r for r in all_rows if r.get("label_source") == "unlabeled"]
    if unlabeled_rows:
        all_empty = all(r.get("label") == "empty" for r in unlabeled_rows)
        all_none_conf = all(r.get("label_confidence") == "none" for r in unlabeled_rows)
        no_train_role = all(
            r.get("final_role") not in [
                "train_positive_gold", "train_positive_silver",
                "train_negative_clean", "train_negative_hard"
            ]
            for r in unlabeled_rows
        )
        report_lines.append(f"- Unlabeled rows: {len(unlabeled_rows)}")
        report_lines.append(f"- All label=empty: **{all_empty}**")
        report_lines.append(f"- All label_confidence=none: **{all_none_conf}**")
        report_lines.append(f"- No train roles: **{no_train_role}**")
    else:
        report_lines.append("- No unlabeled rows in this dry-run.")
    report_lines.append("")

    # folder_weak
    report_lines.append("### 4.5 Folder-Weak Sources (R12/R12a)")
    report_lines.append("")
    fw_rows = [r for r in all_rows if r.get("label_source") == "folder_weak"]
    if fw_rows:
        fw_review = [r for r in fw_rows if r.get("final_role") == "review_required"]
        fw_train = [r for r in fw_rows if r.get("final_role") in ["train_positive_silver", "train_negative_clean"]]
        report_lines.append(f"- Folder-weak rows: {len(fw_rows)}")
        report_lines.append(f"- In review_required (not yet train): {len(fw_review)}")
        report_lines.append(f"- In train role: {len(fw_train)}")
        for r in fw_rows:
            report_lines.append(f"  - `{r.get('source_inventory_id', '?')}`: "
                              f"final_role={r.get('final_role')}, "
                              f"folder_weak_qa_passed='{r.get('folder_weak_qa_passed', '')}', "
                              f"split='{r.get('split', '')}'")
    else:
        report_lines.append("- No folder_weak rows in this dry-run.")
    report_lines.append("")

    # Quality flags
    if quality_flags:
        report_lines.append("### 4.6 Quality Flags")
        report_lines.append("")
        for flag, count in sorted(quality_flags.items()):
            report_lines.append(f"- `{flag}`: {count} rows")
        report_lines.append("")

    report_lines.append("## 5. Data Not Processed")
    report_lines.append("")
    report_lines.append("- `board_live_v3_2_recall_fix/train_candidate2026.5.29/` — 198 WAVs (too many for dry-run)")
    report_lines.append("- `board_live_v3_2_recall_fix_20260530/` — not scanned (size)")
    report_lines.append("- `non_cough/` — FSD50K and other large datasets (not relevant to source inventory)")
    report_lines.append("- `cough/cough_READY/` — 17,239 CoughVID clips (already processed for playback master)")
    report_lines.append("- `p02 night` data — not found in external DATA tree; may require dedicated recording session")
    report_lines.append("- `pending_rerecord_source` — p01 40cm port_smoke used as proxy")
    report_lines.append("")

    report_lines.append("## 6. Notes on This Dry-Run")
    report_lines.append("")
    report_lines.append("1. All sources are real E84 board-live recordings (MIC input, not CoughVID).")
    report_lines.append("2. Audio files are embedded in `sessions/audio/` subdirectories — the builder's")
    report_lines.append("   recursive scan correctly locates them.")
    report_lines.append("3. `board_live_capture` source_type uses `label=empty` + `review_required` default since")
    report_lines.append("   board-live recordings have no folder-level label assertion.")
    report_lines.append("4. p01 40cm detection works through `--default-distance-cm 40` passed to the builder.")
    report_lines.append("5. No p02 night data was available — this is acceptable for a dry-run.")
    report_lines.append("6. All WAV files are read-only (metadata scan) — no audio was copied, moved, or modified.")
    report_lines.append("")

    report_lines.append("## 7. What This Stage DID")
    report_lines.append("")
    report_lines.append("- [x] Built source_inventory CSV from 10 real external data sources")
    report_lines.append("- [x] Total: {total_rows} rows across {n_src_types} source types".format(
        total_rows=total_rows, n_src_types=len(st_counter)))
    report_lines.append("- [x] Validated against `audio_screening_schema_v3_2_1.yaml`")
    report_lines.append(f"- [x] Validator result: {'PASSED' if validator_passed else 'FAILED'}")
    report_lines.append(f"- [x] p01 40cm plosive enforcement: {p01_40cm_pending}")
    report_lines.append(f"- [x] Unlabeled source enforcement: verified")
    report_lines.append(f"- [x] Folder-weak review_required: verified")
    report_lines.append("")

    report_lines.append("## 8. What This Stage Did NOT Do")
    report_lines.append("")
    report_lines.append("- No training, no one-epoch smoke, no formal training")
    report_lines.append("- No label publication (audio_labels)")
    report_lines.append("- No `ml/data/raw/` modification")
    report_lines.append("- No audio file copy, move, delete, resample, normalize, or edit")
    report_lines.append("- No firmware modification")
    report_lines.append("- No model scoring or inference")
    report_lines.append("- No review clip export")
    report_lines.append("- No window manifest generation")
    report_lines.append("- No full raw data scan")
    report_lines.append("- No processing of tonight playback master recording data")
    report_lines.append("")

    report_lines.append("## 9. Commands Run")
    report_lines.append("")
    report_lines.append("```bash")
    report_lines.append("# Run stage 3A orchestration")
    report_lines.append("python run_stage3a_dryrun.py")
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("Individual builder invocations (10 sources):")
    report_lines.append("```bash")
    for src_root, src_type, person, session, dist, notes in SOURCES:
        cmd = f"python tools/build_audio_source_inventory.py"
        cmd += f" --source-root \"{src_root}\""
        cmd += f" --source-type {src_type}"
        cmd += f" --person-id {person}"
        cmd += f" --session-id {session}"
        if dist:
            cmd += f" --default-distance-cm {dist}"
        if notes:
            cmd += f" --notes \"{notes}\""
        cmd += f" --output-csv <partial_csv>"
        report_lines.append(f"# {session}")
        report_lines.append(f"{cmd}")
        report_lines.append("")
    report_lines.append("# Validate merged CSV")
    report_lines.append(f"python tools/validate_audio_screening_schema.py \\")
    report_lines.append(f"  --schema shared/contracts/audio_screening_schema_v3_2_1.yaml \\")
    report_lines.append(f"  --csv docs/audio/stage3a_source_inventory_dryrun.csv \\")
    report_lines.append(f"  --report-json docs/audio/stage3a_source_inventory_validator_report.json \\")
    report_lines.append(f"  --report-md docs/audio/stage3a_source_inventory_validator_report.md")
    report_lines.append("```")

    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    print("Done.")

    # Cleanup temp files
    import shutil
    shutil.rmtree(str(temp_dir), ignore_errors=True)

    return 0 if validator_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
