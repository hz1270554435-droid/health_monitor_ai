from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.common.splits import assign_group_splits


def prepare_radar_splits(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    split_cfg = config["split"]
    target_column = config["training"]["target_column"]
    if target_column not in features.columns:
        raise ValueError(f"Radar training target column not found: {target_column}")
    if "split" in features.columns:
        leaking = features.groupby(split_cfg["group_column"])["split"].nunique(dropna=True)
        leaking = leaking[leaking > 1]
        if not leaking.empty:
            raise ValueError(
                f"{split_cfg['group_column']} appears in multiple splits: "
                f"{sorted(leaking.index.astype(str))}"
            )
        return features.copy()
    return assign_group_splits(
        features,
        group_column=split_cfg["group_column"],
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(config["project"]["seed"]),
    )


def select_feature_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    excluded = set(config.get("features", {}).get("exclude_columns", []))
    excluded.add(config["training"]["target_column"])
    numeric_cols = [
        column
        for column in frame.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded and not frame[column].isna().all()
    ]
    if not numeric_cols:
        raise ValueError("No numeric radar feature columns are available for tree training")
    return numeric_cols


def train_random_forest(features: pd.DataFrame, config: dict[str, Any]) -> tuple[RandomForestClassifier, pd.DataFrame, dict[str, Any]]:
    model_name = str(config["training"].get("model", "random_forest"))
    if model_name != "random_forest":
        raise ValueError(f"Unsupported radar tree model '{model_name}'. First version supports random_forest only.")

    data = prepare_radar_splits(features, config)
    feature_columns = select_feature_columns(data, config)
    target_column = config["training"]["target_column"]
    train_rows = data[data["split"] == "train"].copy()
    eval_rows = data[data["split"].isin(["val", "test"])].copy()
    if train_rows.empty:
        raise ValueError("Radar training split has no rows")
    if eval_rows.empty:
        raise ValueError("Radar evaluation split has no val/test rows")

    medians = train_rows[feature_columns].median(numeric_only=True).fillna(0.0)
    x_train = train_rows[feature_columns].fillna(medians)
    y_train = train_rows[target_column].astype(str)

    model = RandomForestClassifier(
        n_estimators=int(config["training"].get("n_estimators", 100)),
        max_depth=config["training"].get("max_depth"),
        min_samples_leaf=int(config["training"].get("min_samples_leaf", 1)),
        class_weight=config["training"].get("class_weight"),
        random_state=int(config["training"].get("random_state", config["project"]["seed"])),
    )
    model.fit(x_train, y_train)

    predictions = data.copy()
    predictions["pred_radar_state"] = model.predict(data[feature_columns].fillna(medians))
    metrics = {
        "model": model_name,
        "feature_columns": feature_columns,
        "split_counts": data["split"].value_counts().to_dict(),
    }
    for split in ("val", "test"):
        split_rows = predictions[predictions["split"] == split]
        if split_rows.empty:
            continue
        metrics[split] = {
            "rows": int(len(split_rows)),
            "accuracy": float(
                accuracy_score(
                    split_rows[target_column].astype(str),
                    split_rows["pred_radar_state"].astype(str),
                )
            ),
            "classification_report": classification_report(
                split_rows[target_column].astype(str),
                split_rows["pred_radar_state"].astype(str),
                output_dict=True,
                zero_division=0,
            ),
        }
    return model, predictions, metrics


def feature_importance_frame(model: RandomForestClassifier, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_.astype(float),
        }
    ).sort_values("importance", ascending=False)
