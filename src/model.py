"""Model factory for baseline training."""
from __future__ import annotations

# LightGBMを追加
import lightgbm as lgb

# numpyを追加
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, KFold


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]
    fit_intercept: bool = True


# LightGBMに適応
def create_model(config: ModelConfig):
    if config.type == "ridge":
        return Ridge(**config.params)
    if config.type == "linear_regression":
        return LinearRegression(fit_intercept=config.fit_intercept)
    # LightGBMのサポートを追加
    if config.type == "lightgbm":
        # best_paramsにはfit_interceptは含まれないため、fit_intercept以外のパラメータを渡す
        model_params = config.params.copy()
        model_params.pop("fit_intercept", None)
        return lgb.LGBMRegressor(**model_params)
    raise ValueError(f"Unsupported model type: {config.type}")


def build_cv(strategy: str, n_splits: int, shuffle: bool = False, random_state: int | None = None):
    if strategy == "time_series":
        return TimeSeriesSplit(n_splits=n_splits)
    if strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    raise ValueError(f"Unsupported CV strategy: {strategy}")


def rmse(y_true, y_pred):
    # MSEを計算し、その平方根を返す
    return np.sqrt(mean_squared_error(y_true, y_pred))


__all__ = ["ModelConfig", "create_model", "build_cv", "rmse"]
