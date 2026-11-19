
import numpy as np
import pytest

from src.features import build_features
from src.predict import prepare_transaction


def test_prediction_features_match_training():

    training_row = {
        "step": 27,
        "type": "TRANSFER",
        "amount": 150000,
        "is_fraud": 0,
    }

    import pandas as pd

    expected = build_features(
        pd.DataFrame([training_row])
    )

    actual = prepare_transaction(
        amount=150000,
        transaction_type="TRANSFER",
        step=27,
    )

    for column in [
        "amount",
        "log_amount",
        "hour_sin",
        "hour_cos",
    ]:
        assert np.isclose(
            expected[column].iloc[0],
            actual[column].iloc[0],
        )

    assert (
        expected["type"].iloc[0]
        == actual["type"].iloc[0]
    )


def test_negative_amount_rejected():

    with pytest.raises(ValueError):
        prepare_transaction(
            -100,
            "TRANSFER",
            27,
        )


def test_invalid_transaction_type():

    with pytest.raises(ValueError):
        prepare_transaction(
            1000,
            "UNKNOWN",
            27,
        )


def test_fractional_step_rejected():

    with pytest.raises(ValueError):
        prepare_transaction(
            1000,
            "TRANSFER",
            2.5,
        )
