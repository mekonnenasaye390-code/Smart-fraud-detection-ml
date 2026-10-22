"""Leak-free feature engineering as a scikit-learn transformer (runs inside CV)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Time -> cyclic hour features, Amount -> log scale, plus PCA-component norms."""

    def fit(self, X: pd.DataFrame, y=None):
        self.v_cols_ = [c for c in X.columns if c.startswith("V")]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        hour = (X["Time"] // 3600) % 24
        out = X.drop(columns=["Time", "Amount"])
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_night"] = (hour < 6).astype(int)
        out["log_amount"] = np.log1p(X["Amount"].clip(lower=0))
        v = X[self.v_cols_]
        out["v_norm"] = np.sqrt((v ** 2).sum(axis=1))
        out["v_max_abs"] = v.abs().max(axis=1)
        return out
