"""Data loading. Uses the Kaggle 'creditcard.csv' if present, otherwise a synthetic
dataset with the same schema (Time, V1..V28, Amount, Class) and ~0.5% fraud."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from config import DATA_DIR, RANDOM_STATE, TARGET


def make_synthetic(n_samples: int = 60_000, fraud_rate: float = 0.005,
                   seed: int = RANDOM_STATE) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples, n_features=28, n_informative=14, n_redundant=4,
        n_clusters_per_class=2, weights=[1 - fraud_rate, fraud_rate],
        class_sep=1.2, flip_y=0.002, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])

    fraud = y == 1
    hours = rng.uniform(0, 24, n_samples)
    night_fraud = fraud & (rng.random(n_samples) < 0.45)   # fraud skews to night
    hours[night_fraud] = rng.uniform(0, 5, night_fraud.sum())
    day = rng.integers(0, 2, n_samples)
    df["Time"] = day * 86_400 + hours * 3_600

    amount = rng.lognormal(3.2, 1.2, n_samples)
    amount[fraud] = rng.lognormal(4.0, 1.5, fraud.sum())
    df["Amount"] = amount.round(2)
    df[TARGET] = y
    return df.sort_values("Time").reset_index(drop=True)


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        real = DATA_DIR / "creditcard.csv"
        if real.exists():
            path = real
    if path is not None:
        df = pd.read_csv(path)
        print(f"[data] loaded {len(df):,} rows from {path}")
    else:
        DATA_DIR.mkdir(exist_ok=True)
        df = make_synthetic()
        df.to_csv(DATA_DIR / "synthetic_transactions.csv", index=False)
        print(f"[data] no dataset found -> generated synthetic data ({len(df):,} rows)")
    validate(df)
    return df


def validate(df: pd.DataFrame) -> None:
    required = {"Time", "Amount", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Target column must be binary (0/1).")
    rate = df[TARGET].mean()
    print(f"[data] fraud rate = {rate:.4%}  ({int(df[TARGET].sum()):,} frauds)")
