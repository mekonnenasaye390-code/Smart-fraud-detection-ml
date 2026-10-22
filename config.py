"""Central configuration for the Smart Fraud Detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TARGET = "Class"          # 1 = fraud, 0 = legitimate
TEST_SIZE = 0.20
CV_FOLDS = 5

# Business cost model (used to pick the decision threshold).
# Missing a fraud is far more expensive than reviewing a false alert.
COST_FN = 100.0           # cost of a missed fraud (money lost + chargeback)
COST_FP = 5.0             # cost of a false alert (manual review + customer friction)

MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
