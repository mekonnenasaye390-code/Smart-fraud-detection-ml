import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import TARGET
from src.data import make_synthetic
from src.evaluate import choose_threshold, compute_metrics
from src.features import FraudFeatureEngineer
from src.models import build_candidates
from src.monitor import psi


def test_synthetic_schema_and_imbalance():
    df = make_synthetic(n_samples=5000)
    assert {"Time", "Amount", TARGET}.issubset(df.columns)
    assert 0 < df[TARGET].mean() < 0.05


def test_feature_engineering_has_no_raw_time_amount():
    df = make_synthetic(n_samples=1000).drop(columns=[TARGET])
    out = FraudFeatureEngineer().fit_transform(df)
    assert "Time" not in out and "Amount" not in out
    assert {"hour_sin", "hour_cos", "is_night", "log_amount"}.issubset(out.columns)
    assert not out.isna().any().any()


def test_model_beats_random_and_threshold_valid():
    df = make_synthetic(n_samples=8000)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    pipe = build_candidates(0, 100)["hist_gb"].fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    thr = choose_threshold(ytr, pipe.predict_proba(Xtr)[:, 1], 100, 5)
    m = compute_metrics(yte, proba, thr, 100, 5)
    assert m["pr_auc"] > yte.mean() * 3
    assert 0 < thr < 1


def test_psi_zero_for_identical_and_positive_for_shift():
    s = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(s, s) < 1e-6
    assert psi(s, s + 2) > 0.25
