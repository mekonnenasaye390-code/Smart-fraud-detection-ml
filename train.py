"""End-to-end training: CV model selection -> optional tuning -> cost-based threshold
-> hold-out evaluation -> explainability -> saved model bundle.

Usage (from project root):
    python -m src.train                      # synthetic data or data/creditcard.csv
    python -m src.train --data path.csv --tune --n-iter 20
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)

from config import (COST_FN, COST_FP, CV_FOLDS, MODEL_PATH, MODELS_DIR, RANDOM_STATE,
                    REPORTS_DIR, TARGET, TEST_SIZE)
from src.data import load_data
from src.evaluate import choose_threshold, compute_metrics, save_json, save_plots
from src.explain import permutation_report, shap_report
from src.models import build_candidates, search_space


def main(args: argparse.Namespace) -> dict:
    t0 = time.time()
    df = load_data(args.data)
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"[split] train={len(X_tr):,} test={len(X_te):,} pos_weight={pos_weight:.1f}")

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall"}

    # 1) Compare candidates with stratified CV (PR-AUC is the primary metric)
    candidates = build_candidates(RANDOM_STATE, pos_weight)
    rows = []
    for name, pipe in candidates.items():
        s = time.time()
        r = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({"model": name,
                     "pr_auc": r["test_pr_auc"].mean(), "pr_auc_std": r["test_pr_auc"].std(),
                     "roc_auc": r["test_roc_auc"].mean(), "recall@0.5": r["test_recall"].mean(),
                     "sec": time.time() - s})
        print(f"[cv] {name:15s} PR-AUC={rows[-1]['pr_auc']:.4f} ± {rows[-1]['pr_auc_std']:.4f} "
              f"ROC-AUC={rows[-1]['roc_auc']:.4f}  ({rows[-1]['sec']:.0f}s)")
    board = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    board.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = board.loc[0, "model"]
    best = candidates[best_name]
    print(f"[select] best model: {best_name}")

    # 2) Optional hyper-parameter tuning
    if args.tune:
        search = RandomizedSearchCV(best, search_space(best_name), n_iter=args.n_iter,
                                    scoring="average_precision", cv=3, random_state=RANDOM_STATE,
                                    n_jobs=1, refit=True)
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        print(f"[tune] best CV PR-AUC={search.best_score_:.4f}  params={search.best_params_}")

    # 3) Cost-optimal threshold from OUT-OF-FOLD predictions (no test leakage)
    oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thr = choose_threshold(y_tr, oof, COST_FN, COST_FP)
    print(f"[threshold] cost-optimal threshold = {thr:.3f}")

    # 4) Final fit on all training data + untouched hold-out evaluation
    best.fit(X_tr, y_tr)
    proba = best.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, proba, thr, COST_FN, COST_FP)
    metrics["at_default_0.5"] = compute_metrics(y_te, proba, 0.5, COST_FN, COST_FP)
    metrics["model"] = best_name
    save_json(metrics, REPORTS_DIR / "metrics.json")
    save_plots(y_te, proba, thr, COST_FN, COST_FP, REPORTS_DIR)

    # 5) Explainability
    permutation_report(best, X_te, y_te, REPORTS_DIR)
    shap_report(best, X_te, REPORTS_DIR)

    # 6) Persist model bundle
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": best, "threshold": thr, "model_name": best_name,
                 "feature_columns": list(X.columns), "metrics": metrics,
                 "train_reference": X_tr.sample(min(5000, len(X_tr)), random_state=0)},
                MODEL_PATH)

    print("\n=== HOLD-OUT TEST RESULTS ===")
    for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "mcc",
              "tp", "fp", "fn", "tn", "expected_cost", "baseline_cost_no_model", "cost_saving_pct"):
        v = metrics[k]
        print(f"{k:24s} {v:,.4f}" if isinstance(v, float) else f"{k:24s} {v:,}")
    print(f"\n[done] model -> {MODEL_PATH}\n[done] reports -> {REPORTS_DIR}  ({time.time() - t0:.0f}s)")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Smart Fraud Detection model")
    p.add_argument("--data", default=None, help="Path to a CSV (Time, V1..Vn, Amount, Class)")
    p.add_argument("--tune", action="store_true", help="Randomized hyper-parameter search")
    p.add_argument("--n-iter", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
