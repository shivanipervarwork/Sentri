
"""
SENTRI — Step 11: Fraud Detection Model Training

Trains:
    1. Logistic Regression
    2. XGBoost

Uses:
    - Sampled chronological training data
    - Full chronological validation data
    - Training-only preprocessing
    - Class imbalance handling
    - Precision, recall, F1 and average precision
    - Validation-based threshold selection

Does NOT use test.csv.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from xgboost import XGBClassifier


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
METADATA_PATH = DATA_DIR / "split_metadata.json"

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "type",
    "hour_sin",
    "hour_cos",
]

NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "hour_sin",
    "hour_cos",
]

CATEGORICAL_FEATURES = ["type"]

TARGET = "is_fraud"

RANDOM_STATE = 42
CHUNK_SIZE = 50000

# Approximate maximum number of legitimate training rows
MAX_LEGITIMATE_TRAIN = 200000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("sentri.training")


# ==========================================
# 2. LOAD DATASET METADATA
# ==========================================

def load_metadata():
    """Load split statistics created in Step 10."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing metadata: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


# ==========================================
# 3. SAMPLE TRAINING DATA
# ==========================================

def load_training_sample(metadata):
    """
    Keep all fraudulent training rows and randomly
    select legitimate rows to reduce memory usage.
    """

    train_stats = metadata["splits"]["train"]

    legitimate_count = int(
        train_stats["legitimate"]
    )

    if legitimate_count <= 0:
        raise ValueError("No legitimate training records")

    sampling_probability = min(
        1.0,
        MAX_LEGITIMATE_TRAIN / legitimate_count,
    )

    logger.info(
        "Legitimate sampling probability: %.6f",
        sampling_probability,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    selected_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TRAIN_PATH,
            usecols=FEATURE_COLUMNS + [TARGET],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        fraud_mask = chunk[TARGET] == 1

        random_mask = (
            rng.random(len(chunk))
            < sampling_probability
        )

        selected_mask = (
            fraud_mask
            | ((chunk[TARGET] == 0) & random_mask)
        )

        selected = chunk.loc[selected_mask].copy()

        if not selected.empty:
            selected_chunks.append(selected)

        if chunk_number % 20 == 0:
            logger.info(
                "Processed %d training chunks",
                chunk_number,
            )

    if not selected_chunks:
        raise ValueError("Training sample is empty")

    train_df = pd.concat(
        selected_chunks,
        ignore_index=True,
    )

    if train_df[TARGET].nunique() != 2:
        raise ValueError(
            "Training sample must contain both classes"
        )

    return train_df, sampling_probability


# ==========================================
# 4. LOAD VALIDATION DATA
# ==========================================

def load_validation_data():
    """
    Load the full validation period.

    We do not downsample validation data because
    its natural fraud prevalence must be preserved.
    """

    df = pd.read_csv(
        VALIDATION_PATH,
        usecols=FEATURE_COLUMNS + [TARGET],
    )

    if df.empty:
        raise ValueError("Validation dataset is empty")

    if df[TARGET].nunique() != 2:
        raise ValueError(
            "Validation data must contain both classes"
        )

    return df


# ==========================================
# 5. BUILD PREPROCESSING PIPELINES
# ==========================================

def build_preprocessor(scale_numeric: bool):
    """
    Prepare numerical and categorical features.

    Logistic Regression: numerical scaling.
    XGBoost: numerical features passed unchanged.

    Transaction type: one-hot encoding.
    """

    numeric_transformer = (
        StandardScaler()
        if scale_numeric
        else "passthrough"
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


# ==========================================
# 6. BUILD LOGISTIC REGRESSION MODEL
# ==========================================

def build_logistic_model():

    return Pipeline([
        (
            "preprocessing",
            build_preprocessor(
                scale_numeric=True
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# ==========================================
# 7. BUILD XGBOOST MODEL
# ==========================================

def build_xgboost_model(y_train):

    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())

    if fraud_count == 0:
        raise ValueError("No fraud cases in training data")

    class_ratio = legitimate_count / fraud_count

    logger.info(
        "XGBoost scale_pos_weight: %.4f",
        class_ratio,
    )

    return Pipeline([
        (
            "preprocessing",
            build_preprocessor(
                scale_numeric=False
            ),
        ),
        (
            "classifier",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_ratio,
                random_state=RANDOM_STATE,
                n_jobs=4,
            ),
        ),
    ])


# ==========================================
# 8. CHOOSE VALIDATION THRESHOLD
# ==========================================

def choose_threshold(y_true, scores):
    """
    Find the classification threshold that
    maximizes F1 on validation data.
    """

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            scores,
        )
    )

    # The final precision/recall point
    # has no corresponding threshold.
    p = precision[:-1]
    r = recall[:-1]

    f1_values = (
        2 * p * r
        / np.maximum(p + r, 1e-12)
    )

    best_position = int(
        np.argmax(f1_values)
    )

    best_threshold = float(
        thresholds[best_position]
    )

    return best_threshold


# ==========================================
# 9. EVALUATE A TRAINED MODEL
# ==========================================

def evaluate_model(
    model,
    X_validation,
    y_validation,
):
    """Evaluate on the full validation dataset."""

    scores = model.predict_proba(
        X_validation
    )[:, 1]

    threshold = choose_threshold(
        y_validation,
        scores,
    )

    predictions = (
        scores >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "average_precision": float(
            average_precision_score(
                y_validation,
                scores,
            )
        ),
        "precision": float(
            precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )
        ),
        "threshold": threshold,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts_generated": int(
            predictions.sum()
        ),
    }

    return metrics


# ==========================================
# 10. SAVE MODEL
# ==========================================

def save_model(model, name):
    """Save preprocessing and classifier together."""

    path = MODEL_DIR / f"{name}.joblib"

    joblib.dump(
        model,
        path,
        compress=3,
    )

    logger.info("Saved model: %s", path)


# ==========================================
# 11. MAIN TRAINING WORKFLOW
# ==========================================

def main():

    logger.info("Starting Sentri model training")

    metadata = load_metadata()

    train_df, sampling_probability = (
        load_training_sample(metadata)
    )

    validation_df = load_validation_data()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]

    X_validation = validation_df[FEATURE_COLUMNS]
    y_validation = validation_df[TARGET]

    logger.info(
        "Sampled training rows: %s",
        f"{len(train_df):,}",
    )

    logger.info(
        "Full validation rows: %s",
        f"{len(validation_df):,}",
    )

    logger.info(
        "Fraudulent training rows: %d",
        int(y_train.sum()),
    )

    logger.info(
        "Fraudulent validation rows: %d",
        int(y_validation.sum()),
    )

    models = {
        "logistic_regression": build_logistic_model(),
        "xgboost": build_xgboost_model(y_train),
    }

    results = {}

    for model_name, model in models.items():

        logger.info(
            "Training model: %s",
            model_name,
        )

        model.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Evaluating model: %s",
            model_name,
        )

        metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        results[model_name] = metrics

        logger.info(
            "%s | AP: %.4f | Precision: %.4f | "
            "Recall: %.4f | F1: %.4f",
            model_name,
            metrics["average_precision"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
        )

        save_model(
            model,
            model_name,
        )

    # Select by threshold-independent average precision
    best_model = max(
        results,
        key=lambda name: results[name][
            "average_precision"
        ],
    )

    report = {
        "project": "Sentri",
        "evaluation_dataset": "validation",
        "training_method": (
            "All training fraud cases plus "
            "randomly sampled legitimate cases"
        ),
        "random_state": RANDOM_STATE,
        "training_rows_used": int(
            len(train_df)
        ),
        "training_fraud_count": int(
            y_train.sum()
        ),
        "validation_rows": int(
            len(validation_df)
        ),
        "validation_fraud_count": int(
            y_validation.sum()
        ),
        "legitimate_sampling_probability": float(
            sampling_probability
        ),
        "best_model_by_average_precision": best_model,
        "model_results": results,
    }

    report_path = (
        REPORT_DIR / "validation_metrics.json"
    )

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info(
        "Best model by validation AP: %s",
        best_model,
    )

    logger.info(
        "Saved validation report: %s",
        report_path,
    )

    logger.info(
        "Sentri model training completed!"
    )


if __name__ == "__main__":
    main()
