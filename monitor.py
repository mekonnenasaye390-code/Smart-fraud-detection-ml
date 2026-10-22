"""Data-drift monitoring with the Population Stability Index (PSI).

    python -m src.monitor --new data/new_transactions.csv
PSI < 0.1 stable | 0.1-0.25 moderate drift | > 0.25 major drift (retrain)
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from config import MODEL_PATH


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.dropna(), edges)[0] / max(len(expected.dropna()), 1)
    a = np.histogram(actual.dropna(), edges)[0] / max(len(actual.dropna()), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = [{"feature": c, "psi": psi(reference[c], new[c])}
            for c in reference.columns if c in new.columns]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    df["status"] = pd.cut(df["psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "moderate", "major"])
    return df


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--new", required=True)
    a = p.parse_args()
    bundle = joblib.load(MODEL_PATH)
    rep = drift_report(bundle["train_reference"], pd.read_csv(a.new))
    print(rep.head(15).to_string(index=False))
    if (rep["psi"] > 0.25).any():
        print("\n[ALERT] Major drift detected -> consider retraining.")


if __name__ == "__main__":
    main()
