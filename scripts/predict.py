"""Generate Kaggle submission predictions using trained fold models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

# Ensure src modules are importable when running as a script
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features import FeatureConfig, SimpleFeatureExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.seed is not None:
        config.data["seed"] = args.seed
    config.ensure_dirs()

    paths = config.get("paths")
    files = config.get("files")
    target_cfg = config.get("target")
    cv_cfg = config.get("cv")
    feature_cfg = config.get("features")
    inference_cfg = config.get("inference")

    test_df = pd.read_csv(Path(paths["input_dir"]) / files["test"])
    sample_submission = pd.read_csv(Path(paths["input_dir"]) / files["sample_submission"])

    feature_config = FeatureConfig(
        drop_columns=feature_cfg.get("drop_columns", []),
        imputation_strategy=feature_cfg.get("imputation_strategy", "median"),
        scale=feature_cfg.get("scale", True),
        rolling_windows=feature_cfg.get("rolling_windows"),
        enable_interactions=feature_cfg.get("enable_interactions", False),
        time_column=cv_cfg.get("time_column"),
        group_column=cv_cfg.get("group_column"),
    )
    extractor = SimpleFeatureExtractor(feature_config)

    model_files = sorted(Path(paths["models_dir"]).glob(f"{config.get('run_name')}_fold_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained fold models found. Run scripts/train.py first.")

    fold_predictions: List[np.ndarray] = []
    for model_file in model_files:
        artifact = joblib.load(model_file)
        model = artifact["model"]
        scaler = artifact.get("scaler")
        feature_columns = artifact["feature_columns"]

        extractor.scaler = scaler
        transformed = extractor.transform(test_df, target_column=target_cfg.get("column"))
        # Align feature columns if the test set lost any columns during processing.
        missing_cols = [col for col in feature_columns if col not in transformed.columns]
        for col in missing_cols:
            transformed[col] = 0.0
        transformed = transformed[feature_columns]
        preds = model.predict(transformed)
        fold_predictions.append(preds)
        print(f"Loaded model from {model_file}")

    if inference_cfg.get("average_folds", True):
        predictions = np.mean(fold_predictions, axis=0)
    else:
        predictions = fold_predictions[-1]

    submission = sample_submission.copy()
    submission[target_cfg.get("prediction_column", target_cfg.get("column"))] = predictions

    output_filename = inference_cfg.get("output_filename", f"{config.get('run_name')}_submission.csv")
    submission_path = Path(paths["submissions_dir"]) / output_filename
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    meta = {
        "run_name": config.get("run_name"),
        "models": [str(p) for p in model_files],
        "submission_path": str(submission_path),
    }
    meta_path = Path(paths["logs_dir"]) / f"{config.get('run_name')}_inference.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
