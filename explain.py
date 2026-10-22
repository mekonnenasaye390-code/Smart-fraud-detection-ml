"""Model explainability: permutation importance (always) + SHAP (if installed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_report(pipe, X, y, out_dir: Path, top: int = 15, seed: int = 42) -> pd.DataFrame:
    res = permutation_importance(pipe, X, y, scoring="average_precision",
                                 n_repeats=3, random_state=seed, n_jobs=1)
    df = (pd.DataFrame({"feature": X.columns, "importance": res.importances_mean,
                        "std": res.importances_std})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "feature_importance.csv", index=False)
    t = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(t["feature"], t["importance"], xerr=t["std"], color="#2563eb")
    ax.set(xlabel="Drop in PR-AUC when shuffled", title="Permutation importance (test set)")
    fig.tight_layout(); fig.savefig(out_dir / "feature_importance.png", dpi=150); plt.close(fig)
    return df


def shap_report(pipe, X, out_dir: Path, sample: int = 1000) -> bool:
    """SHAP summary for tree models (RandomForest / LightGBM). Returns True on success."""
    try:
        import shap
        clf = pipe.named_steps["clf"]
        if clf.__class__.__name__ not in {"RandomForestClassifier", "LGBMClassifier"}:
            return False
        Xf = pipe.named_steps["fe"].transform(X.sample(min(sample, len(X)), random_state=0))
        values = shap.TreeExplainer(clf).shap_values(Xf)
        if isinstance(values, list):
            values = values[1]
        elif getattr(values, "ndim", 2) == 3:
            values = values[:, :, 1]
        shap.summary_plot(values, Xf, show=False)
        plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=150); plt.close()
        return True
    except Exception as e:  # SHAP is optional
        print(f"[explain] SHAP skipped: {e}")
        return False
