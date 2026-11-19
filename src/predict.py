
"""
SENTRI — Reusable Fraud Prediction

Loads the winning model and frozen validation threshold.
Scores transactions without retraining the model.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "validation_metrics.json"
)

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "type",
    "hour_sin",
    "hour_cos",
]

ALLOWED_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
}


# ------------------------------------------
# 1. SELECT MODEL USING VALIDATION RESULTS
# ------------------------------------------

def get_model_selection():
    """Read the previously saved model comparison."""

    with REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    results = report["model_results"]

    winner = report["best_model_by_average_precision"]

    if winner not in results:
        raise ValueError(
            "Selected model missing from validation report."
        )

    threshold = float(
        results[winner]["threshold"]
    )

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")

    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")

    return winner, threshold


# ------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------

@lru_cache(maxsize=2)
def load_model(model_name: str):
    """Load a trusted, locally trained model once."""

    allowed = {
        "logistic_regression",
        "xgboost",
    }

    if model_name not in allowed:
        raise ValueError("Unknown model name")

    model_path = MODEL_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    return joblib.load(model_path)


# ------------------------------------------
# 3. SCORE AN ALREADY ENGINEERED DATAFRAME
# ------------------------------------------

def score_features(
    feature_df: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    """Return fraud risk scores for prepared features."""

    missing = set(FEATURE_COLUMNS) - set(
        feature_df.columns
    )

    if missing:
        raise ValueError(
            f"Missing model features: {sorted(missing)}"
        )

    model = load_model(model_name)

    scores = model.predict_proba(
        feature_df[FEATURE_COLUMNS]
    )[:, 1]

    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Model returned invalid scores")

    return scores


# ------------------------------------------
# 4. PREPARE A NEW UNLABELED TRANSACTION
# ------------------------------------------

def prepare_transaction(
    amount: float,
    transaction_type: str,
    step: int,
) -> pd.DataFrame:
    """
    Generate exactly the five baseline features.

    No true fraud label is required.
    """

    amount = float(amount)
    step = float(step)

    transaction_type = transaction_type.upper().strip()

    if not np.isfinite(amount) or amount < 0:
        raise ValueError(
            "Amount must be finite and non-negative"
        )

    if not np.isfinite(step) or step < 1:
        raise ValueError("Invalid simulation step")

    if not step.is_integer():
        raise ValueError(
            "Simulation step must be a whole number"
        )

    if transaction_type not in ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported type: {transaction_type}"
        )

    hour = int(step) % 24
    angle = 2 * np.pi * hour / 24

    return pd.DataFrame([{
        "amount": amount,
        "log_amount": np.log1p(amount),
        "type": transaction_type,
        "hour_sin": np.sin(angle),
        "hour_cos": np.cos(angle),
    }])


# ------------------------------------------
# 5. PUBLIC FRAUD PREDICTION FUNCTION
# ------------------------------------------

def predict_transaction(
    amount: float,
    transaction_type: str,
    step: int,
) -> dict:
    """Score one transaction using the chosen model."""

    model_name, threshold = get_model_selection()

    features = prepare_transaction(
        amount=amount,
        transaction_type=transaction_type,
        step=step,
    )

    score = float(
        score_features(features, model_name)[0]
    )

    flagged = score >= threshold

    return {
        "model": model_name,
        "amount": float(amount),
        "type": transaction_type.upper().strip(),
        "step": int(step),
        "risk_score": round(score, 6),
        "threshold": threshold,
        "flagged_for_review": bool(flagged),
        "decision": (
            "FLAG FOR REVIEW"
            if flagged
            else "NOT FLAGGED"
        ),
    }


# ------------------------------------------
# 6. LOCAL DEMONSTRATION
# ------------------------------------------

if __name__ == "__main__":

    result = predict_transaction(
        amount=150000,
        transaction_type="TRANSFER",
        step=27,
    )

    print(json.dumps(result, indent=4))
