"""Candidate models + hyper-parameter search spaces."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FraudFeatureEngineer

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False


def build_candidates(seed: int, pos_weight: float) -> dict:
    c = {
        "logreg": Pipeline([
            ("fe", FraudFeatureEngineer()), ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "random_forest": Pipeline([
            ("fe", FraudFeatureEngineer()),
            ("clf", RandomForestClassifier(
                n_estimators=150, min_samples_leaf=2, n_jobs=-1,
                class_weight="balanced_subsample", random_state=seed)),
        ]),
        "hist_gb": Pipeline([
            ("fe", FraudFeatureEngineer()),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=300, early_stopping=True,
                class_weight="balanced", random_state=seed)),
        ]),
        "hist_gb_smote": ImbPipeline([
            ("fe", FraudFeatureEngineer()),
            ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=300, early_stopping=True, random_state=seed)),
        ]),
    }
    if HAS_LGBM:
        c["lightgbm"] = Pipeline([
            ("fe", FraudFeatureEngineer()),
            ("clf", LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8,
                scale_pos_weight=min(pos_weight, 100), random_state=seed,
                verbose=-1, n_jobs=-1)),
        ])
    return c


def search_space(name: str) -> dict:
    if name == "logreg":
        return {"clf__C": loguniform(1e-3, 1e2)}
    if name == "random_forest":
        return {"clf__max_depth": [None, 8, 12, 20], "clf__min_samples_leaf": randint(1, 8),
                "clf__max_features": ["sqrt", "log2", 0.3]}
    if name == "lightgbm":
        return {"clf__num_leaves": randint(15, 80), "clf__learning_rate": loguniform(1e-2, 2e-1),
                "clf__min_child_samples": randint(10, 60), "clf__reg_lambda": loguniform(1e-2, 1e1)}
    # hist_gb / hist_gb_smote
    return {"clf__learning_rate": loguniform(1e-2, 2e-1), "clf__max_leaf_nodes": randint(15, 63),
            "clf__min_samples_leaf": randint(10, 60), "clf__l2_regularization": uniform(0, 5)}
