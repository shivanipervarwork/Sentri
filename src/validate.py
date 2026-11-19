
import numpy as np
import pandas as pd


# Required columns in the PaySim dataset
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


# Allowed PaySim transaction types
VALID_TYPES = {
    "PAYMENT",
    "TRANSFER",
    "CASH_OUT",
    "CASH_IN",
    "DEBIT",
}


# Rename CSV columns to match PostgreSQL
COLUMN_MAPPING = {
    "nameOrig": "name_orig",
    "oldbalanceOrg": "oldbalance_org",
    "newbalanceOrig": "newbalance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "oldbalance_dest",
    "newbalanceDest": "newbalance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
}


# Balance-related columns
BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def validate_chunk(df: pd.DataFrame):
    """
    Validate a batch of PaySim transactions.

    Returns:
        clean: DataFrame containing valid transactions
        rejected_count: Number of rejected rows
        rejection_counts: Validation failures by reason
    """

    # STEP 1: Check required columns
    required = REQUIRED_COLUMNS + [
        "source_file",
        "source_row_number",
    ]

    missing_columns = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Make a copy so we don't modify the original data
    df = df[required].copy()

    # STEP 2: Convert numeric columns
    numeric_columns = [
        "step",
        "amount",
        "isFraud",
        "isFlaggedFraud",
    ] + BALANCE_COLUMNS

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # STEP 3: Validate account identifiers
    valid_accounts = (
        df["nameOrig"].notna()
        & df["nameDest"].notna()
        & df["nameOrig"].astype("string")
          .str.strip().ne("").fillna(False)
        & df["nameDest"].astype("string")
          .str.strip().ne("").fillna(False)
    )

    # STEP 4: Validate balance columns
    valid_balances = pd.Series(
        np.isfinite(
            df[BALANCE_COLUMNS].to_numpy(dtype=float)
        ).all(axis=1),
        index=df.index,
    )

    # STEP 5: Define validation checks
    checks = {
        "invalid_step": (
            df["step"].notna()
            & np.isfinite(df["step"])
            & (df["step"] >= 1)
            & (df["step"] % 1 == 0)
        ),

        "invalid_amount": (
            df["amount"].notna()
            & np.isfinite(df["amount"])
            & (df["amount"] >= 0)
        ),

        "invalid_type": (
            df["type"].isin(VALID_TYPES)
        ),

        "invalid_account": valid_accounts,

        "invalid_fraud_label": (
            df["isFraud"].isin([0, 1])
        ),

        "invalid_flag_label": (
            df["isFlaggedFraud"].isin([0, 1])
        ),

        "invalid_balance": valid_balances,

        "invalid_source": (
            df["source_file"].notna()
            & df["source_file"].astype("string")
              .str.strip().ne("").fillna(False)
            & pd.to_numeric(
                df["source_row_number"],
                errors="coerce",
            ).notna()
        ),
    }

    # STEP 6: Combine validation results
    valid_mask = pd.Series(
        True,
        index=df.index,
    )

    rejection_counts = {}

    for reason, condition in checks.items():
        condition = condition.fillna(False)

        rejection_counts[reason] = int(
            (~condition).sum()
        )

        valid_mask &= condition

    # STEP 7: Count rejected transactions
    rejected_count = int(
        (~valid_mask).sum()
    )

    # STEP 8: Keep valid transactions
    clean = df.loc[valid_mask].copy()

    # STEP 9: Convert integer columns
    integer_columns = [
        "step",
        "isFraud",
        "isFlaggedFraud",
        "source_row_number",
    ]

    for column in integer_columns:
        clean[column] = clean[column].astype("int64")

    # STEP 10: Rename columns for PostgreSQL
    clean = clean.rename(
        columns=COLUMN_MAPPING
    )

    # STEP 11: Return results
    return clean, rejected_count, rejection_counts
