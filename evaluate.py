"""Metrics, cost-sensitive threshold selection and report plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def _cost(y, proba, thr, cost_fn, cost_fp):
    pred = proba >= thr
    fn = int(((~pred) & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return fn * cost_fn + fp * cost_fp


def cost_curve(y, proba, cost_fn, cost_fp, grid=None):
    grid = (np.unique(np.concatenate([np.logspace(-4, -1, 60), np.linspace(0.1, 0.995, 120)]))
            if grid is None else grid)
    return grid, np.array([_cost(y, proba, t, cost_fn, cost_fp) for t in grid])


def choose_threshold(y, proba, cost_fn, cost_fp) -> float:
    grid, costs = cost_curve(np.asarray(y), proba, cost_fn, cost_fp)
    return float(grid[int(np.argmin(costs))])


def compute_metrics(y, proba, thr, cost_fn, cost_fp) -> dict:
    y = np.asarray(y)
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cost = fn * cost_fn + fp * cost_fp
    baseline = int(y.sum()) * cost_fn          # no model: every fraud is missed
    return {
        "threshold": round(float(thr), 4),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "expected_cost": float(cost), "baseline_cost_no_model": float(baseline),
        "cost_saving_pct": float(100 * (1 - cost / baseline)) if baseline else 0.0,
        "alerts_per_10k_transactions": float(10_000 * pred.mean()),
    }


def save_plots(y, proba, thr, cost_fn, cost_fp, out_dir: Path, tag: str = "test") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y)

    p, r, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(r, p, lw=2); ax.axhline(y.mean(), ls="--", c="gray", label="Random")
    ax.set(xlabel="Recall", ylabel="Precision",
           title=f"Precision-Recall (AP={average_precision_score(y, proba):.3f})")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / f"{tag}_pr_curve.png", dpi=150); plt.close(fig)

    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2); ax.plot([0, 1], [0, 1], "--", c="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title=f"ROC (AUC={roc_auc_score(y, proba):.3f})")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_roc_curve.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y, (proba >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"],
           yticklabels=["Legit", "Fraud"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix @ threshold {thr:.3f}")
    fig.tight_layout(); fig.savefig(out_dir / f"{tag}_confusion_matrix.png", dpi=150); plt.close(fig)

    grid, costs = cost_curve(y, proba, cost_fn, cost_fp)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(grid, costs, lw=2); ax.axvline(thr, c="red", ls="--", label=f"chosen = {thr:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Total cost", title="Cost vs. threshold"); ax.set_xscale("symlog", linthresh=0.1)
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / f"{tag}_cost_curve.png", dpi=150); plt.close(fig)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
