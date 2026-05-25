from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.common.config import load_config
from src.common.io import ensure_dir
from v3_1_mining_common import audio_info, display_path, file_sha1, safe_token, stable_id


MANIFEST_COLUMNS = [
    "audio_file",
    "source_domain",
    "session_id",
    "person_id",
    "scenario_hint",
    "duration_sec",
    "sample_rate",
    "channels",
    "file_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the v3.1 external audio mining manifest.")
    parser.add_argument("--config", default="configs/audio_mining_v3_1.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-files", type=int, default=None, help="Optional limiter for smoke runs.")
    parser.add_argument("--require-sources", action="store_true", help="Fail if any configured source is missing.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def source_extensions(source: dict[str, Any]) -> set[str]:
    values = source.get("extensions") or [".wav", ".pcm", ".raw"]
    return {str(value).lower() if str(value).startswith(".") else f".{str(value).lower()}" for value in values}


def iter_audio_files(source_root: Path, extensions: set[str]) -> list[Path]:
    return sorted(path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def generic_sample_dir(name: str) -> bool:
    return bool(re.match(r"^(cough|non_cough|sample|clip)[_-]?\d+$", name.strip().lower()))


def relative_parts(path: Path, source_root: Path) -> tuple[str, ...]:
    try:
        return path.parent.resolve().relative_to(source_root.resolve()).parts
    except ValueError:
        return ()


def infer_scenario_hint(path: Path, source_root: Path, fallback: str) -> str:
    parts = [part for part in relative_parts(path, source_root) if part not in {"", "."}]
    informative = [part for part in parts if not generic_sample_dir(part)]
    if informative:
        return safe_token(informative[-1], 80)
    return str(fallback)


def infer_session_id(path: Path, source_root: Path, source_domain: str) -> str:
    parts = relative_parts(path, source_root)
    session_key = "/".join(parts) if parts else path.stem
    return stable_id("session", source_domain, session_key, length=14)


def infer_person_id(path: Path, source_root: Path, source_domain: str) -> str:
    parts = relative_parts(path, source_root)
    if len(parts) >= 2:
        person_key = parts[0]
    elif len(parts) == 1 and not generic_sample_dir(parts[0]):
        person_key = parts[0]
    else:
        person_key = "/".join(parts) if parts else path.stem
    return stable_id("person", source_domain, person_key, length=14)


def build_rows(config: dict[str, Any], max_files: int | None, require_sources: bool) -> tuple[list[dict[str, object]], dict[str, Any]]:
    mining = config.get("mining", {})
    pcm_defaults = dict(mining.get("pcm_defaults", {}))
    sources = list(mining.get("sources", []))
    missing_policy = str(mining.get("missing_source_policy", "warn")).strip().lower()
    rows: list[dict[str, object]] = []
    missing_sources: list[str] = []
    errors: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    files_seen = 0

    for source in sources:
        source_domain = str(source["source_domain"])
        source_root = resolve_path(source["path"])
        if not source_root.exists():
            missing_sources.append(str(source_root))
            if require_sources or missing_policy == "error":
                raise FileNotFoundError(f"Configured v3.1 mining source does not exist: {source_root}")
            continue

        extensions = source_extensions(source)
        for audio_path in iter_audio_files(source_root, extensions):
            if max_files is not None and files_seen >= max_files:
                break
            files_seen += 1
            try:
                info = audio_info(audio_path, {**pcm_defaults, **dict(source.get("pcm", {}))})
                rows.append(
                    {
                        "audio_file": display_path(audio_path),
                        "source_domain": source_domain,
                        "session_id": infer_session_id(audio_path, source_root, source_domain),
                        "person_id": infer_person_id(audio_path, source_root, source_domain),
                        "scenario_hint": infer_scenario_hint(
                            audio_path,
                            source_root,
                            str(source.get("scenario_hint", source_domain)),
                        ),
                        "duration_sec": info["duration_sec"],
                        "sample_rate": info["sample_rate"],
                        "channels": info["channels"],
                        "file_hash": file_sha1(audio_path),
                    }
                )
                source_counts[source_domain] = source_counts.get(source_domain, 0) + 1
            except Exception as exc:
                errors.append({"audio_file": str(audio_path), "error": str(exc)})
        if max_files is not None and files_seen >= max_files:
            break

    report = {
        "rows": len(rows),
        "source_counts": source_counts,
        "missing_sources": missing_sources,
        "error_rows": len(errors),
        "errors": errors[:50],
    }
    return rows, report


def main() -> int:
    args = parse_args()
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive when provided")

    config = load_config(args.config)
    out_path = resolve_path(args.out or config["paths"]["manifest_csv"])
    report_path = out_path.with_name(f"{out_path.stem}_report.json")
    rows, report = build_rows(config, args.max_files, args.require_sources)
    ensure_dir(out_path.parent)
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(out_path, index=False)
    report.update({"manifest_csv": str(out_path), "report_json": str(report_path)})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("COPY_FOR_CODEX_BEGIN")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("COPY_FOR_CODEX_END")
    return 0 if not report["error_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
