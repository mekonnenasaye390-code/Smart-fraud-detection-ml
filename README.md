# Smart Fraud Detection (pure ML, no frontend)

Production-style credit-card fraud detection pipeline built for an AI/ML internship portfolio.

## What makes it "advanced"
- **Extreme class imbalance** (~0.5% fraud): class weights, SMOTE inside the CV pipeline (no leakage), PR-AUC as the primary metric
- **Model bake-off** with stratified 5-fold CV: Logistic Regression, Random Forest, HistGradientBoosting, HistGB + SMOTE, LightGBM (auto-used if installed)
- **Optional hyper-parameter tuning** (`--tune`, RandomizedSearchCV on PR-AUC)
- **Cost-sensitive decision threshold** (missed fraud = 100, false alert = 5), chosen on out-of-fold predictions so the test set stays untouched
- **Explainability**: permutation importance (+ SHAP for tree models if installed)
- **Drift monitoring** with PSI, **batch/single inference CLI**, **unit tests**

## Setup
```bash
pip install -r requirements.txt
python -m src.train                 # synthetic data, or put Kaggle creditcard.csv in data/
python -m src.train --tune --n-iter 20
python -m pytest -q
```
Dataset: Kaggle "Credit Card Fraud Detection" (`creditcard.csv`, columns Time, V1..V28, Amount, Class) -> place in `data/`. Without it, a synthetic dataset with the same schema is generated.

## Use the model
```bash
python -m src.predict --input data/synthetic_transactions.csv --output reports/scored.csv
python -m src.monitor --new data/new_transactions.csv
```

## Outputs
`models/fraud_model.joblib`, `reports/metrics.json`, `model_comparison.csv`, PR/ROC/confusion/cost curves, `feature_importance.png`.

## Structure
```
config.py          settings & cost model
src/data.py        loading + synthetic generator
src/features.py    leak-free feature engineering transformer
src/models.py      candidates + search spaces
src/train.py       full training pipeline
src/evaluate.py    metrics, threshold selection, plots
src/explain.py     permutation importance / SHAP
src/predict.py     inference CLI
src/monitor.py     PSI drift monitoring
tests/             pytest suite
```
