
"""
SENTRI — Feature Engineering

Transforms PaySim transactions into machine-learning
features without using fraud labels or future information.
"""

import numpy as np
import pandas as pd


# Features used by our baseline ML models
FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "type",
    "hour_sin",
    "hour_cos",
]

TARGET_COLUMN = "is_fraud"

OUTPUT_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw transactions into ML-ready features.

    Parameters:
        df: DataFrame containing source transactions.

    Returns:
        DataFrame containing five features and the target.
    """

    # Step 1: Verify required columns exist
    required_columns = [
        "step",
        "type",
        "amount",
        "is_fraud",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Step 2: Copy the required source columns
    data = df[required_columns].copy()

    # Step 3: Validate numeric values
    data["step"] = pd.to_numeric(
        data["step"], errors="raise"
    )

    data["amount"] = pd.to_numeric(
        data["amount"], errors="raise"
    )

    data["is_fraud"] = pd.to_numeric(
        data["is_fraud"], errors="raise"
    )

    if not np.isfinite(data["amount"]).all():
        raise ValueError("Non-finite transaction amount")

    if (data["amount"] < 0).any():
        raise ValueError("Negative transaction amount")

    if not np.isfinite(data["step"]).all():
        raise ValueError("Invalid simulation step")

    if (
        (data["step"] < 1)
        | (data["step"] % 1 != 0)
    ).any():
        raise ValueError("Simulation step must be positive integer")

    if not data["is_fraud"].isin([0, 1]).all():
        raise ValueError("Invalid fraud labels")

    # Step 4: Create log-transformed amount
    data["log_amount"] = np.log1p(
        data["amount"]
    )

    # Step 5: Extract hour of simulation day
    hour = data["step"] % 24

    # Step 6: Create circular time features
    angle = 2 * np.pi * hour / 24

    data["hour_sin"] = np.sin(angle)
    data["hour_cos"] = np.cos(angle)

    # Step 7: Preserve transaction type as text
    data["type"] = data["type"].astype(str)

    # Step 8: Convert target to integer
    data["is_fraud"] = data["is_fraud"].astype(int)

    # Step 9: Return only allowed features and target
    return data[OUTPUT_COLUMNS].copy()
