
import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    build_features,
)


def sample_transaction(
    amount=1000,
    step=24,
    fraud=0,
):
    return pd.DataFrame([{
        "step": step,
        "type": "TRANSFER",
        "amount": amount,
        "is_fraud": fraud,
    }])


def test_feature_columns():

    result = build_features(sample_transaction())

    assert list(result.columns) == (
        FEATURE_COLUMNS + ["is_fraud"]
    )


def test_log_amount():

    result = build_features(
        sample_transaction(amount=1000)
    )

    expected = np.log1p(1000)

    assert np.isclose(
        result["log_amount"].iloc[0],
        expected,
    )


def test_circular_hour():

    result = build_features(
        sample_transaction(step=24)
    )

    assert np.isclose(
        result["hour_sin"].iloc[0],
        0,
        atol=1e-10,
    )

    assert np.isclose(
        result["hour_cos"].iloc[0],
        1,
        atol=1e-10,
    )


def test_negative_amount_rejected():

    with pytest.raises(ValueError):
        build_features(
            sample_transaction(amount=-50)
        )


def test_invalid_fraud_label_rejected():

    with pytest.raises(ValueError):
        build_features(
            sample_transaction(fraud=3)
        )


def test_no_leakage_columns():

    df = sample_transaction()

    df["is_flagged_fraud"] = 1
    df["newbalance_orig"] = 0

    result = build_features(df)

    assert "is_flagged_fraud" not in result.columns
    assert "newbalance_orig" not in result.columns


def test_original_dataframe_unchanged():

    original = sample_transaction()
    before = original.copy(deep=True)

    build_features(original)

    pd.testing.assert_frame_equal(
        original,
        before,
    )
