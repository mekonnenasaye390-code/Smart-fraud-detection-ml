# Smart Fraud Detection — Portfolio Case Study (paste into your portfolio site)

**Role:** Solo ML engineer · **Stack:** Python, scikit-learn, imbalanced-learn, LightGBM (optional), pandas, matplotlib

## Problem
Card fraud is rare (~0.5% of transactions) but costly. Accuracy is meaningless here, and missing a fraud costs far more than a false alert.

## Approach
1. Leak-free feature engineering (cyclic hour, log amount, PCA-component norms) inside the sklearn pipeline
2. Compared 4–5 models with stratified 5-fold CV using PR-AUC; handled imbalance with class weights and SMOTE
3. Chose the decision threshold by minimizing business cost on out-of-fold predictions
4. Evaluated once on an untouched hold-out set; explained the model with permutation importance
5. Added PSI drift monitoring, an inference CLI and unit tests

## Results (hold-out set, synthetic data — replace with your Kaggle run)
- ROC-AUC ≈ 0.95, PR-AUC ≈ 0.70, recall ≈ 73%
- ≈ 66% lower expected fraud cost than having no model

## What I learned
Metric choice and threshold selection matter more than the algorithm on imbalanced data; resampling must happen inside cross-validation to avoid leakage.
