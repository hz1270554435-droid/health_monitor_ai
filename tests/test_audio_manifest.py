from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio.manifest import build_audio_manifest, validate_manifest


def test_build_audio_manifest_from_label_and_hard_data_sources(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "\n".join(
            [
                "clip_id,session_id,person_id,start_time,end_time,label,audio_file,split,hard_negative",
                "clip_cough,S1,P1,0.0,1.0,cough,audio/cough.wav,train,False",
                "clip_non_cough,S2,P2,0.0,1.0,non_cough,audio/non_cough.wav,val,True",
            ]
        ),
        encoding="utf-8",
    )
    hard_manifest = tmp_path / "hard_data_manifest.csv"
    hard_manifest.write_text(
        "\n".join(
            [
                "source_audio,copied_audio,duration,cumulative_duration",
                "D:/external/noise.wav,data/hard/noise.wav,1.25,1.25",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "paths": {
            "audio_manifest_csv": "data/index/audio_manifest.csv",
            "non_cough_pool_csv": "data/index/non_cough_pool.csv",
        },
        "manifest": {
            "sources": {
                "label_csvs": [
                    {
                        "name": "unit_labels",
                        "path": str(labels),
                        "audio_root": ".",
                        "source_dataset": "unit",
                    }
                ],
                "hard_data_manifests": [
                    {
                        "name": "unit_hard",
                        "path": str(hard_manifest),
                        "source_dataset": "external_negative",
                        "label": "non_cough",
                        "eval_allowed": False,
                    }
                ],
            }
        },
    }

    manifest = build_audio_manifest(config, root=tmp_path)

    assert len(manifest) == 3
    assert set(manifest["normalized_label"]) == {"cough", "non_cough"}
    assert manifest.loc[manifest["source_dataset"] == "external_negative", "split"].iloc[0] == "train"
    validate_manifest(manifest)
