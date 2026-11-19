
import pandas as pd

from src.validate import validate_chunk


def make_transaction(amount=1000, fraud=0):
    """Create one sample transaction for testing."""

    return pd.DataFrame([{
        "step": 1,
        "type": "TRANSFER",
        "amount": amount,
        "nameOrig": "C100",
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 4000,
        "nameDest": "C200",
        "oldbalanceDest": 2000,
        "newbalanceDest": 3000,
        "isFraud": fraud,
        "isFlaggedFraud": 0,
        "source_file": "paysim.csv",
        "source_row_number": 1,
    }])


def test_valid_transaction():
    """A valid transaction should pass validation."""

    df = make_transaction()

    clean, rejected, reasons = validate_chunk(df)

    assert len(clean) == 1
    assert rejected == 0


def test_negative_amount():
    """A negative transaction amount must be rejected."""

    df = make_transaction(amount=-500)

    clean, rejected, reasons = validate_chunk(df)

    assert len(clean) == 0
    assert rejected == 1
    assert reasons["invalid_amount"] == 1


def test_invalid_fraud_label():
    """A fraud label other than 0 or 1 must be rejected."""

    df = make_transaction(fraud=5)

    clean, rejected, reasons = validate_chunk(df)

    assert len(clean) == 0
    assert rejected == 1
    assert reasons["invalid_fraud_label"] == 1
