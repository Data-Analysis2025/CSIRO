"""Train a htmp-style baseline model (used for CSIRO Biomass)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Ensure src modules are importable when running as a script
import sys


# OptunaとLightGBM
import lightgbm as lgb
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.features import FeatureConfig, SimpleFeatureExtractor
from src.model import ModelConfig, build_cv, create_model, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--skip-optuna", action="store_true", help="Skip Optuna search and use config params as-is.")
    return parser.parse_args()


# ヘルパー関数
def parse_optuna_params(trial: optuna.Trial, param_config: dict) -> dict:
    params = {}
    for name, config in param_config.items():
        if isinstance(config, dict):
            low = config["low"]
            high = config["high"]

            if config["type"] == "loguniform":
                params[name] = trial.suggest_float(name, float(low), float(high), log=True)
            elif config["type"] == "uniform":
                params[name] = trial.suggest_float(name, float(low), float(high))
            elif config["type"] == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif config["type"] == "float":
                params[name] = trial.suggest_float(name, float(low), float(high))
        
        else:
            params[name] = config
    return params


def resolve_param_defaults(param_config: dict) -> dict:
    """When Optuna is skipped, pick a deterministic value from search spaces."""
    params = {}
    for name, config in param_config.items():
        if isinstance(config, dict):
            low = config["low"]
            high = config["high"]
            if config["type"] == "loguniform":
                params[name] = float(np.exp((np.log(float(low)) + np.log(float(high))) / 2.0))
            elif config["type"] in ("uniform", "float"):
                params[name] = float((float(low) + float(high)) / 2.0)
            elif config["type"] == "int":
                params[name] = int(round((int(low) + int(high)) / 2))
            else:
                params[name] = config
        else:
            params[name] = config
    return params


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
    model_cfg = config.get("model")
    optuna_cfg = config.get("optuna") # Optuna設定を読み込む

    train_df = pd.read_csv(Path(paths["input_dir"]) / files["train"])
    target_column = target_cfg["column"]
    if target_column not in train_df.columns:
        raise KeyError(f"Target column '{target_column}' not found in training data.")

    y = train_df[target_column].values

    feature_config = FeatureConfig(
        drop_columns=feature_cfg.get("drop_columns", []),
        imputation_strategy=feature_cfg.get("imputation_strategy", "median"),
        imputation_rolling_windows=feature_cfg.get("imputation_rolling_windows"),
        imputation_rolling_weights=feature_cfg.get("imputation_rolling_weights"),
        scale=feature_cfg.get("scale", True),
        rolling_windows=feature_cfg.get("rolling_windows"),
        rolling_stats=feature_cfg.get("rolling_stats"),
        enable_interactions=feature_cfg.get("enable_interactions", False),
        time_column=cv_cfg.get("time_column"),
        group_column=cv_cfg.get("group_column"),
    )
    extractor = SimpleFeatureExtractor(feature_config)
    X = extractor.fit_transform(train_df, target_column=target_column)

    cv_strategy = build_cv(
        strategy=cv_cfg.get("strategy", "time_series"),
        n_splits=cv_cfg.get("n_splits", 5),
        shuffle=cv_cfg.get("strategy") != "time_series",
        random_state=config.get("seed"),
    )


    # Optunaによるハイパーパラメータ探索（省略可）
    if args.skip_optuna or optuna_cfg is None or optuna_cfg.get("n_trials", 0) <= 0:
        study = None
        best_params = resolve_param_defaults(model_cfg.get("params", {}))
        print("Optuna skipped: using deterministic defaults from config search space.")
    else:
        def objective(trial: optuna.Trial) -> float:
            model_params = parse_optuna_params(trial, model_cfg.get("params", {}))
            if model_cfg.get("type", "lightgbm") == "lightgbm":
                model_params.setdefault("n_estimators", 100)  # Epoch=100相当
                model_params.setdefault("device", "gpu")      # GPU優先

            scores = []
            cv_search = build_cv(
                strategy=cv_cfg.get("strategy", "time_series"),
                n_splits=cv_cfg.get("n_splits", 5),
                shuffle=cv_cfg.get("strategy") != "time_series",
                random_state=config.get("seed"),
            )

            for fold, (train_idx, valid_idx) in enumerate(cv_search.split(X, y)):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                search_model = ModelConfig(
                    type=model_cfg.get("type", "lightgbm"),
                    params=model_params,
                    fit_intercept=model_cfg.get("fit_intercept", False),
                )
                model = create_model(search_model)
                fit_kwargs = {}
                if isinstance(model, lgb.LGBMRegressor):
                    fit_kwargs = {
                        "eval_set": [(X_valid, y_valid)],
                        "eval_metric": "rmse",
                        "callbacks": [lgb.early_stopping(100, verbose=False)],
                    }
                model.fit(X_train, y_train, **fit_kwargs)
                preds = model.predict(X_valid)
                scores.append(rmse(y_valid, preds))

            mean_rmse = np.mean(scores)
            print(f"Trial {trial.number}: mean CV RMSE={mean_rmse:.5f}")
            return mean_rmse

        study = optuna.create_study(direction=optuna_cfg.get("direction", "minimize"))
        n_trials = optuna_cfg.get("n_trials", 50)
        trial_bar = tqdm(total=n_trials, desc="Optuna trials", unit="trial")
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[lambda _study, _trial: trial_bar.update(1)],
            )
        finally:
            trial_bar.close()

        best_params = study.best_params
        print(f"--- Optuna Search Finished ---")
        print(f"Best RMSE (avg): {study.best_value:.5f}")
        print(f"Best Hyperparameters: {best_params}")


    # 見つけたBest ParamsでK-Fold訓練 & モデル保存
    # model_configを、Optunaで見つけたbest_paramsで上書きする
    final_model_params = model_cfg.get("params", {}).copy()
    final_model_params.update(best_params) # best_paramsで上書き
    if model_cfg.get("type", "lightgbm") == "lightgbm":
        final_model_params.setdefault("n_estimators", 100)
        final_model_params.setdefault("device", "gpu")

    model_config = ModelConfig(
        type=model_cfg.get("type", "lightgbm"),
        params=final_model_params, # Optunaの結果を使う
        fit_intercept=model_cfg.get("fit_intercept", False),
    )

    oof_predictions = np.zeros(len(train_df))
    scores: List[float] = []

    fold_iter = cv_strategy.split(X, y)
    total_folds = cv_cfg.get("n_splits", 5)
    for fold, (train_idx, valid_idx) in enumerate(
        tqdm(fold_iter, total=total_folds, desc="Final CV", unit="fold")
    ):
        model = create_model(model_config)
        # LGBMの場合のみ early_stopping を使用
        fit_kwargs = {}
        if isinstance(model, lgb.LGBMRegressor):
            fit_kwargs = {
                "eval_set": [(X.iloc[valid_idx], y[valid_idx])],
                "eval_metric": "rmse",
                "callbacks": [lgb.early_stopping(100, verbose=False)],
            }
        model.fit(X.iloc[train_idx], y[train_idx], **fit_kwargs)
        preds = model.predict(X.iloc[valid_idx])
        oof_predictions[valid_idx] = preds
        fold_score = rmse(y[valid_idx], preds)
        scores.append(fold_score)
        model_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_fold_{fold}.pkl"
        joblib.dump(
            {
                "model": model,
                "scaler": extractor.scaler,
                "feature_columns": list(X.columns),
            },
            model_path,
        )
        print(f"Fold {fold}: RMSE={fold_score:.5f}")

    overall_rmse = rmse(y, oof_predictions)
    print(f"Overall OOF RMSE: {overall_rmse:.5f}")

    metadata = {
        "run_name": config.get("run_name"),
        "scores": scores,
        "oof_rmse": overall_rmse,
        "optuna_best_value": study.best_value if study is not None else None,
        "optuna_best_params": best_params,
        "feature_columns": list(X.columns),
        "config_path": str(config.path),
    }
    metadata_path = Path(paths["models_dir"]) / f"{config.get('run_name')}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    oof_path = Path(paths["processed_dir"]) / f"{config.get('run_name')}_oof.csv"
    Path(paths["processed_dir"]).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"oof_pred": oof_predictions}).to_csv(oof_path, index=False)


if __name__ == "__main__":
    main()
