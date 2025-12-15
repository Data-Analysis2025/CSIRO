"""Feature engineering utilities shared across htmp-style baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    drop_columns: Sequence[str]
    imputation_strategy: str = "median"
    scale: bool = True
    rolling_windows: Sequence[int] | None = None
    rolling_stats: Sequence[str] | None = None
    imputation_rolling_windows: Sequence[int] | None = None
    imputation_rolling_weights: Sequence[float] | None = None
    enable_interactions: bool = False
    time_column: str | None = None
    group_column: str | None = None


class SimpleFeatureExtractor:
    """Extracts simple statistical features suitable for the HTMP dataset."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler: StandardScaler | None = None
        self.numeric_columns: List[str] = []

    def _prepare_columns(
        self, df: pd.DataFrame, target_column: str | None, fit: bool
    ) -> List[str]:
        drop_cols = set(self.config.drop_columns)
        if target_column is not None:
            drop_cols.add(target_column)

        if fit or not self.numeric_columns:
            numeric_columns = [
                col
                for col in df.columns
                if col not in drop_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
            if not numeric_columns:
                raise ValueError("No numeric columns available for feature extraction.")
            self.numeric_columns = numeric_columns
        numeric_columns = [col for col in self.numeric_columns if col in df.columns]
        if not numeric_columns:
            raise ValueError("Configured numeric columns are missing from the provided dataframe.")
        return numeric_columns

    def fit_transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=True)

    def transform(self, df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
        return self._create_features(df.copy(), target_column=target_column, fit=False)

    def _create_features(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        numeric_columns = self._prepare_columns(df, target_column=target_column, fit=fit)
        df_numeric = df[numeric_columns].copy()

        df_numeric = self._apply_imputation(df_numeric, df)
        df_numeric = self._add_rolling_statistics(df_numeric, df)
        df_numeric = self._add_interactions(df_numeric)

        if self.config.scale:
            if fit:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(df_numeric)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler has not been fitted. Call fit_transform first.")
                scaled = self.scaler.transform(df_numeric)
            df_numeric = pd.DataFrame(scaled, columns=df_numeric.columns, index=df_numeric.index)

        return df_numeric

    def _apply_imputation(self, df_numeric: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        if self.config.imputation_strategy == "rolling_ensemble":
            return self._rolling_ensemble_impute(df_numeric, df_original)
        # Optuna:ffill->bfill->median
        if self.config.imputation_strategy == "ffill_bfill_median":
            df_filled = df_numeric.ffill()
            df_filled = df_filled.bfill()
            return df_filled.fillna(df_filled.median())

        if self.config.imputation_strategy == "mean":
            return df_numeric.fillna(df_numeric.mean())
        if self.config.imputation_strategy == "median":
            return df_numeric.fillna(df_numeric.median())
        return df_numeric.fillna(0.0)

    def _rolling_ensemble_impute(self, df_numeric: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs using a weighted ensemble of short/medium/long moving averages."""
        windows = list(self.config.imputation_rolling_windows or [3, 7, 21])
        weights = self.config.imputation_rolling_weights
        if weights is None:
            weight_arr = np.ones(len(windows), dtype=float)
        else:
            if len(weights) != len(windows):
                raise ValueError("Length of imputation_rolling_weights must match imputation_rolling_windows.")
            weight_arr = np.asarray(weights, dtype=float)
        weight_arr = weight_arr / weight_arr.sum()

        time_col = self.config.time_column
        group_col = self.config.group_column
        if time_col is None or time_col not in df_original.columns:
            # Fallback if no time column is available
            df_filled = df_numeric.ffill().bfill()
            return df_filled.fillna(df_numeric.median())

        filled = df_numeric.copy()
        for col in df_numeric.columns:
            if group_col and group_col in df_original.columns:
                grouped = df_numeric[col].groupby(df_original[group_col])
                roll_means = [
                    grouped.transform(lambda s, w=window: s.rolling(window=w, min_periods=1).mean())
                    for window in windows
                ]
            else:
                sorter = df_original[time_col].argsort()
                sorted_series = df_numeric[col].iloc[sorter]
                roll_means = [
                    sorted_series.rolling(window=window, min_periods=1).mean().sort_index()
                    for window in windows
                ]

            roll_df = pd.concat(roll_means, axis=1)
            roll_df.columns = [f"w{w}" for w in windows]
            weight_df = pd.DataFrame(
                np.tile(weight_arr, (len(roll_df), 1)),
                index=roll_df.index,
                columns=roll_df.columns,
            )

            weighted_sum = (roll_df * weight_df).sum(axis=1)
            weight_used = weight_df.where(~roll_df.isna()).sum(axis=1)
            ensemble = weighted_sum / weight_used.replace({0: np.nan})
            filled[col] = df_numeric[col].fillna(ensemble)
            filled[col] = filled[col].fillna(df_numeric[col].median())
        return filled

    def _add_rolling_statistics(self, df_numeric: pd.DataFrame, df_original: pd.DataFrame) -> pd.DataFrame:
        windows = self.config.rolling_windows or []
        stats = [s.lower() for s in (self.config.rolling_stats or ["mean"])]
        time_col = self.config.time_column
        group_col = self.config.group_column
        if not windows or time_col is None or time_col not in df_original.columns:
            return df_numeric

        supported = {"mean", "std", "min", "max", "zscore"}
        unknown = set(stats) - supported
        if unknown:
            raise ValueError(f"Unsupported rolling_stats: {unknown}")

        def roll_stat(series: pd.Series, window: int, stat: str) -> pd.Series:
            roll = series.rolling(window=window, min_periods=1)
            if stat == "mean":
                return roll.mean()
            if stat == "std":
                return roll.std(ddof=0)
            if stat == "min":
                return roll.min()
            if stat == "max":
                return roll.max()
            if stat == "zscore":
                mean = roll.mean()
                std = roll.std(ddof=0)
                std_safe = std.replace({0: np.nan})
                return ((series - mean) / std_safe).fillna(0.0)
            raise ValueError(stat)

        df_with_time = df_original[[time_col]].copy()
        if group_col and group_col in df_original.columns:
            df_with_time[group_col] = df_original[group_col]

        augmented = df_numeric.copy()
        for window in windows:
            if group_col and group_col in df_with_time.columns:
                for col in df_numeric.columns:
                    grouped = df_numeric[col].groupby(df_with_time[group_col])
                    for stat in stats:
                        rolled = grouped.transform(lambda s, w=window, st=stat: roll_stat(s, w, st))
                        augmented[f"{col}_roll{window}_{stat}"] = rolled
            else:
                sorter = df_with_time[time_col].argsort()
                df_sorted = df_numeric.iloc[sorter]
                for col in df_numeric.columns:
                    for stat in stats:
                        rolled_values = roll_stat(df_sorted[col], window, stat).sort_index()
                        augmented[f"{col}_roll{window}_{stat}"] = rolled_values
        return augmented

    def _add_interactions(self, df_numeric: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enable_interactions:
            return df_numeric
        interaction_cols = {}
        for i, col_a in enumerate(self.numeric_columns):
            for col_b in self.numeric_columns[i + 1 :]:
                interaction_cols[f"{col_a}_x_{col_b}"] = df_numeric[col_a] * df_numeric[col_b]
        if interaction_cols:
            df_numeric = pd.concat([df_numeric, pd.DataFrame(interaction_cols, index=df_numeric.index)], axis=1)
        return df_numeric


__all__ = ["FeatureConfig", "SimpleFeatureExtractor"]
