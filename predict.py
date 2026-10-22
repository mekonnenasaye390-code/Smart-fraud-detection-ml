"""Batch / single-transaction scoring with the saved model.

    python -m src.predict --input data/synthetic_transactions.csv --output reports/scored.csv
    python -m src.predict --json '{"Time": 3600, "Amount": 250.0, "V1": 0.1, ...}'
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from config import MODEL_PATH, TARGET

_BUNDLE = None


def load_bundle(path=MODEL_PATH) -> dict:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def score_transactions(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    cols = bundle["feature_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    proba = bundle["pipeline"].predict_proba(df[cols])[:, 1]
    thr = bundle["threshold"]
    out = df.copy()
    out["fraud_probability"] = proba.round(6)
    out["is_fraud"] = (proba >= thr).astype(int)
    out["risk_level"] = pd.cut(proba, [-1, thr * 0.5, thr, 2], labels=["low", "medium", "high"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input"); p.add_argument("--output", default="reports/scored.csv")
    p.add_argument("--json", help="Single transaction as a JSON string")
    a = p.parse_args()
    if a.json:
        res = score_transactions(pd.DataFrame([json.loads(a.json)]))
        print(res[["fraud_probability", "is_fraud", "risk_level"]].to_dict("records")[0])
        return
    if not a.input:
        p.error("provide --input or --json")
    df = pd.read_csv(a.input).drop(columns=[TARGET], errors="ignore")
    res = score_transactions(df)
    res.to_csv(a.output, index=False)
    print(f"[predict] {len(res):,} rows scored, {int(res.is_fraud.sum()):,} flagged -> {a.output}")


if __name__ == "__main__":
    main()
