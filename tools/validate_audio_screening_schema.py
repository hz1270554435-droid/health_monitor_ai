#!/usr/bin/env python3
"""
v3.2.1 Audio Screening Schema Validator
========================================
Minimal machine-checkable CSV validator for the v3.2.1 screening schema.

Usage:
  python ml/tools/validate_audio_screening_schema.py \\
    --schema shared/contracts/audio_screening_schema_v3_2_1.yaml \\
    --csv docs/audio/fixtures/v3_2_1_screening_valid_fixture.csv \\
    --report-json docs/audio/fixtures/valid_report.json \\
    --report-md docs/audio/fixtures/valid_report.md

Stage: 2B — fixture smoke only.
No training, no label publication, no data/raw changes, no firmware changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# --- YAML loading (stdlib-only fallback) ---
try:
    import yaml  # type: ignore[import-untyped]
    def load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
except ImportError:
    def load_yaml(path: Path) -> dict[str, Any]:
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        )


# ROOT: find workspace root by walking up from this file until we find
# shared/contracts/ or CLAUDE.md. Because ml/ is a junction, the resolved
# path points to the ML repo; we check known workspace paths as fallback.
def _find_workspace_root() -> Path:
    import os
    # Known workspace roots (check explicitly — junction hides CLAUDE.md)
    known_workspaces = [
        Path(r"d:\e84_health_monitor"),
        Path(r"D:\e84_health_monitor"),
    ]
    for ws in known_workspaces:
        if (ws / "CLAUDE.md").exists() or (ws / "shared" / "contracts").exists():
            return ws
    # Fallback: walk up from __file__
    script = Path(__file__)
    if not script.is_absolute():
        script = Path.cwd() / script
    for candidate in [script] + list(script.parents):
        if (candidate / "CLAUDE.md").exists() or (candidate / "shared" / "contracts").exists():
            return candidate
    # Last resort
    return script.parents[1]


ROOT = _find_workspace_root()


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


# ── helpers ──────────────────────────────────────────────────────────

def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Return (rows, fieldnames) from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def cell(value: str) -> str:
    """Normalize cell value for comparison."""
    return value.strip()


# ── validation context ────────────────────────────────────────────────

class ValidationContext:
    """Holds schema, rows, and accumulates errors/warnings."""

    def __init__(self, schema: dict[str, Any], rows: list[dict[str, str]], columns: list[str]):
        self.schema = schema
        self.rows = rows
        self.columns = columns
        self.errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.warnings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._row_errors: dict[int, set[str]] = defaultdict(set)

    def add_error(self, rule_id: str, detail: dict[str, Any]) -> None:
        self.errors[rule_id].append(detail)
        row_idx = detail.get("row_index", -1)
        if row_idx >= 0:
            self._row_errors[row_idx].add(rule_id)

    def add_warning(self, rule_id: str, detail: dict[str, Any]) -> None:
        self.warnings[rule_id].append(detail)

    def rows_with_errors(self) -> int:
        return len(self._row_errors)

    def total_errors(self) -> int:
        return sum(len(v) for v in self.errors.values())

    def total_warnings(self) -> int:
        return sum(len(v) for v in self.warnings.values())


# ── rule implementations ──────────────────────────────────────────────

def check_required_columns(ctx: ValidationContext) -> None:
    """Check that all required columns exist in the CSV."""
    required = ctx.schema.get("required_columns", [])
    for col in required:
        if col not in ctx.columns:
            ctx.add_error("REQUIRED_COLUMNS", {
                "column": col,
                "row_index": -1,
                "value": "",
                "detail": f"Required column '{col}' is missing from CSV.",
            })


def check_enum(ctx: ValidationContext, rule: dict[str, Any]) -> None:
    """Check that a column's values are all in the enum."""
    column = rule["check"]["column"]
    enum_name = rule["check"]["in_enum"]
    allowed = set(ctx.schema["enums"][enum_name]["values"])
    rule_id = rule["id"]

    if column not in ctx.columns:
        return  # already caught by required_columns

    for i, row in enumerate(ctx.rows):
        value = cell(row.get(column, ""))
        if value == "" and "" in allowed:
            continue
        if value not in allowed:
            ctx.add_error(rule_id, {
                "row_index": i,
                "window_id": row.get("window_id", f"row_{i}"),
                "column": column,
                "value": value,
                "allowed": sorted(allowed),
                "detail": f"Row {i}: '{column}' value '{value}' not in allowed set: {sorted(allowed)}",
            })


def check_cross_column_condition(ctx: ValidationContext, rule: dict[str, Any]) -> None:
    """Check cross-column validation rules with a condition → requirement pattern."""
    chk = rule.get("check", {})
    rule_id = rule["id"]
    severity = rule.get("severity", "error")

    condition = chk.get("condition")
    conditions_all = chk.get("condition_all")
    requires = chk.get("require")
    requires_all = chk.get("require_all")
    forbids_all = chk.get("forbid_all")
    for_each = chk.get("for_each")

    # for_each: each pair defines a constraint. When a row's final_role matches
    # the pair's final_role, label must match the pair's label (or label_not).
    if for_each:
        for pair in for_each:
            role_col = None
            role_val = None
            label_col = None
            label_val = None
            label_not_val = None
            for k, v in pair.items():
                if k.endswith("_not"):
                    # e.g. label_not -> column=label, not_value=v
                    label_col = k.replace("_not", "")
                    label_not_val = v
                elif k == "final_role":
                    role_col = k
                    role_val = v
                elif k == "label":
                    label_col = k
                    label_val = v
                else:
                    # Treat as column=value constraint
                    pass

            for i, row in enumerate(ctx.rows):
                if role_col and role_val and cell(row.get(role_col, "")) != role_val:
                    continue  # pair doesn't apply to this row
                if label_not_val and label_col:
                    if cell(row.get(label_col, "")) == label_not_val:
                        ctx.add_error(rule_id, {
                            "row_index": i,
                            "window_id": row.get("window_id", f"row_{i}"),
                            "detail": f"Row {i}: {label_col} must NOT be '{label_not_val}' for {role_col}={role_val}",
                        })
                elif label_val is not None and label_col:
                    if cell(row.get(label_col, "")) != label_val:
                        ctx.add_error(rule_id, {
                            "row_index": i,
                            "window_id": row.get("window_id", f"row_{i}"),
                            "detail": f"Row {i}: {label_col} must be '{label_val}' for {role_col}={role_val}, got '{cell(row.get(label_col, ''))}'",
                        })
        return

    # Determine which rows are affected
    def row_matches_condition(row: dict[str, str]) -> bool:
        if conditions_all:
            for c in conditions_all:
                col = c["column"]
                val = cell(row.get(col, ""))
                if "equals" in c:
                    if val != c["equals"]:
                        return False
                elif "in_list" in c:
                    if val not in c["in_list"]:
                        return False
                elif "not_in_list" in c:
                    if val in c["not_in_list"]:
                        return False
                elif "not_equals" in c:
                    if val == c["not_equals"]:
                        return False
                else:
                    return False
            return True
        if condition:
            col = condition["column"]
            op = next((k for k in ["equals", "not_equals"] if k in condition), "equals")
            expected = condition[op]
            val = cell(row.get(col, ""))
            return val == expected if op == "equals" else val != expected
        return True  # no condition → all rows

    affected_rows = [
        (i, row) for i, row in enumerate(ctx.rows)
        if row_matches_condition(row)
    ]

    # Check required column values
    if requires:
        col = requires["column"]
        allowed_list = requires.get("in_list")
        not_allowed_list = requires.get("not_in_list")
        equals_val = requires.get("equals")
        not_equals_val = requires.get("not_equals")
        for i, row in affected_rows:
            val = cell(row.get(col, ""))
            ok = True
            if allowed_list is not None:
                ok = ok and val in allowed_list
            if not_allowed_list is not None:
                ok = ok and val not in not_allowed_list
            if equals_val is not None:
                ok = ok and val == equals_val
            if not_equals_val is not None:
                ok = ok and val != not_equals_val
            if not ok:
                detail = f"Row {i}: {col}={val} violates constraint"
                (ctx.add_error if severity == "error" else ctx.add_warning)(
                    rule_id, {"row_index": i, "window_id": row.get("window_id", f"row_{i}"), "detail": detail}
                )

    # Check required_all constraints
    if requires_all:
        for i, row in affected_rows:
            for req in requires_all:
                col = req["column"]
                val = cell(row.get(col, ""))
                ok = True
                if "equals" in req:
                    ok = val == req["equals"]
                elif "not_equals" in req:
                    ok = val != req["not_equals"]
                elif "in_list" in req:
                    ok = val in req["in_list"]
                elif "not_in_list" in req:
                    ok = val not in req["not_in_list"]
                if not ok:
                    detail = f"Row {i}: {col}={val} violates {req}"
                    (ctx.add_error if severity == "error" else ctx.add_warning)(
                        rule_id, {"row_index": i, "window_id": row.get("window_id", f"row_{i}"), "detail": detail}
                    )

    # Check forbid_all constraints
    if forbids_all:
        for i, row in affected_rows:
            for forbid in forbids_all:
                col = forbid["column"]
                val = cell(row.get(col, ""))
                ok = True
                if "equals" in forbid:
                    ok = val != forbid["equals"]
                elif "in_list" in forbid:
                    ok = val not in forbid["in_list"]
                if not ok:
                    detail = f"Row {i}: {col}={val} is forbidden"
                    (ctx.add_error if severity == "error" else ctx.add_warning)(
                        rule_id, {"row_index": i, "window_id": row.get("window_id", f"row_{i}"), "detail": detail}
                    )


def check_group_integrity(ctx: ValidationContext, rule: dict[str, Any]) -> None:
    """Check that all rows sharing a group_by key have the same value for require_same column."""
    chk = rule.get("check", {})
    group_by = chk["group_by"]
    require_same = chk["require_same"]
    rule_id = rule["id"]
    severity = rule.get("severity", "error")

    if group_by not in ctx.columns or require_same not in ctx.columns:
        return

    groups: dict[str, set[str]] = defaultdict(set)
    group_rows: dict[str, list[int]] = defaultdict(list)

    for i, row in enumerate(ctx.rows):
        key = cell(row.get(group_by, ""))
        val = cell(row.get(require_same, ""))
        if key == "":
            continue
        groups[key].add(val)
        group_rows[key].append(i)

    for key, vals in groups.items():
        if len(vals) > 1:
            detail = (
                f"{group_by}='{key}' has multiple {require_same} values: {sorted(vals)}. "
                f"Rows: {group_rows[key]}"
            )
            entry = {
                "row_index": group_rows[key][0],
                "window_id": f"{group_by}={key}",
                "detail": detail,
            }
            (ctx.add_error if severity == "error" else ctx.add_warning)(rule_id, entry)


def _row_has_train_role(row: dict[str, str], train_roles: list[str]) -> bool:
    return cell(row.get("final_role", "")) in train_roles


def _row_in_train(row: dict[str, str]) -> bool:
    return cell(row.get("split", "")) == "train"


def check_all_rules(ctx: ValidationContext) -> None:
    """Run all rules from the schema against the CSV."""
    schema = ctx.schema
    rules = schema.get("rules", [])
    train_roles = schema.get("train_roles", [])

    # Phase 1: Required columns
    check_required_columns(ctx)

    # Phase 2: Per-rule checks
    for rule in rules:
        chk = rule.get("check", {})
        check_type = chk.get("type", "enum")

        if check_type == "enum" or "in_enum" in chk:
            check_enum(ctx, rule)
        elif check_type == "cross_column":
            check_cross_column_condition(ctx, rule)
        elif check_type == "group_integrity":
            check_group_integrity(ctx, rule)
        else:
            # Try to infer from structure
            if "in_enum" in chk:
                check_enum(ctx, rule)
            elif "group_by" in chk:
                check_group_integrity(ctx, rule)
            else:
                check_cross_column_condition(ctx, rule)


# ── report generation ─────────────────────────────────────────────────

def generate_report(ctx: ValidationContext, csv_path: str, schema_path: str, passed: bool) -> dict[str, Any]:
    """Generate the JSON report."""
    error_count_by_rule: dict[str, int] = {}
    warning_count_by_rule: dict[str, int] = {}
    error_examples: dict[str, list[dict[str, Any]]] = {}
    warning_examples: dict[str, list[dict[str, Any]]] = {}

    schema = ctx.schema
    rules_map: dict[str, str] = {}
    for rule in schema.get("rules", []):
        rules_map[rule["id"]] = rule.get("description", "")

    max_examples = schema.get("validator", {}).get("max_error_examples_per_rule", 5)

    for rule_id, items in ctx.errors.items():
        error_count_by_rule[rule_id] = len(items)
        error_examples[rule_id] = items[:max_examples]

    for rule_id, items in ctx.warnings.items():
        warning_count_by_rule[rule_id] = len(items)
        warning_examples[rule_id] = items[:max_examples]

    return {
        "schema_name": schema.get("schema_name", ""),
        "schema_version": schema.get("version", ""),
        "csv_file": csv_path,
        "csv_rows": len(ctx.rows),
        "status": "passed" if passed else "failed",
        "total_errors": ctx.total_errors(),
        "total_warnings": ctx.total_warnings(),
        "rows_with_errors": ctx.rows_with_errors(),
        "error_count_by_rule": error_count_by_rule,
        "warning_count_by_rule": warning_count_by_rule,
        "error_examples": error_examples,
        "warning_examples": warning_examples,
        "rules_descriptions": rules_map,
        "not_executed": [
            "training",
            "one_epoch_smoke",
            "final_label_publication",
            "ml_data_raw_write",
            "firmware_modification",
            "long_full_data_scan",
        ],
    }


def generate_markdown(report: dict[str, Any]) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        "# v3.2.1 Screening Schema Validation Report",
        "",
        f"**Status**: `{report['status'].upper()}`",
        f"**Schema**: `{report['schema_name']}` v{report['schema_version']}",
        f"**CSV**: `{report['csv_file']}`",
        f"**Rows**: {report['csv_rows']}",
        f"**Total Errors**: {report['total_errors']}",
        f"**Total Warnings**: {report['total_warnings']}",
        f"**Rows with Errors**: {report['rows_with_errors']}",
        "",
        "---",
        "",
    ]

    # Error summary
    if report["total_errors"] > 0:
        lines.append("## Errors")
        lines.append("")
        lines.append("| Rule | Count | Description |")
        lines.append("|------|-------|-------------|")
        for rule_id, count in sorted(report["error_count_by_rule"].items()):
            desc = report["rules_descriptions"].get(rule_id, "")
            lines.append(f"| `{rule_id}` | {count} | {desc} |")
        lines.append("")

        lines.append("### Error Examples")
        lines.append("")
        for rule_id, examples in sorted(report["error_examples"].items()):
            if not examples:
                continue
            lines.append(f"**{rule_id}**: {report['rules_descriptions'].get(rule_id, '')}")
            lines.append("")
            for ex in examples[:3]:
                lines.append(f"- `{ex.get('window_id', '')}`: {ex.get('detail', '')}")
            lines.append("")
    else:
        lines.append("## Errors")
        lines.append("")
        lines.append("None. All checks passed.")
        lines.append("")

    # Warning summary
    if report["total_warnings"] > 0:
        lines.append("## Warnings")
        lines.append("")
        lines.append("| Rule | Count | Description |")
        lines.append("|------|-------|-------------|")
        for rule_id, count in sorted(report["warning_count_by_rule"].items()):
            desc = report["rules_descriptions"].get(rule_id, "")
            lines.append(f"| `{rule_id}` | {count} | {desc} |")
        lines.append("")

        lines.append("### Warning Examples")
        lines.append("")
        for rule_id, examples in sorted(report["warning_examples"].items()):
            if not examples:
                continue
            lines.append(f"**{rule_id}**: {report['rules_descriptions'].get(rule_id, '')}")
            lines.append("")
            for ex in examples[:3]:
                lines.append(f"- `{ex.get('window_id', '')}`: {ex.get('detail', '')}")
            lines.append("")
    else:
        lines.append("## Warnings")
        lines.append("")
        lines.append("None.")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Scope",
        "",
        "- Stage 2B: fixture smoke only.",
        "- No training, no label publication, no `ml/data/raw/` changes, no firmware changes.",
        "- This validator does not consume model scores, read audio files, or run full-data scans.",
        "",
        "## Not Executed",
        "",
    ])
    for item in report.get("not_executed", []):
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


# ── main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v3.2.1 Audio Screening Schema Validator — Stage 2B fixture smoke"
    )
    p.add_argument(
        "--schema",
        default="shared/contracts/audio_screening_schema_v3_2_1.yaml",
        help="Path to YAML schema file.",
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file to validate.",
    )
    p.add_argument(
        "--report-json",
        help="Path to write JSON report.",
    )
    p.add_argument(
        "--report-md",
        help="Path to write Markdown report.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    schema_path = resolve_path(args.schema)
    csv_path = resolve_path(args.csv)

    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}", file=sys.stderr)
        return 1
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    schema = load_yaml(schema_path)
    rows, columns = read_csv_rows(csv_path)

    ctx = ValidationContext(schema, rows, columns)
    check_all_rules(ctx)

    passed = ctx.total_errors() == 0
    report = generate_report(ctx, str(csv_path), str(schema_path), passed)

    # Print summary to stdout
    print(json.dumps({
        "status": "passed" if passed else "failed",
        "rows": len(rows),
        "total_errors": ctx.total_errors(),
        "total_warnings": ctx.total_warnings(),
        "rows_with_errors": ctx.rows_with_errors(),
        "error_rules": sorted(ctx.errors.keys()),
        "warning_rules": sorted(ctx.warnings.keys()),
    }, indent=2))

    # Write reports if requested
    if args.report_json:
        rj_path = resolve_path(args.report_json)
        rj_path.parent.mkdir(parents=True, exist_ok=True)
        rj_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  -> JSON report: {rj_path}")

    if args.report_md:
        rm_path = resolve_path(args.report_md)
        rm_path.parent.mkdir(parents=True, exist_ok=True)
        rm_path.write_text(generate_markdown(report), encoding="utf-8")
        print(f"  -> Markdown report: {rm_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
