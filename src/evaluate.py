
"""
SENTRI — Final Held-Out Test Evaluation

IMPORTANT:
- Model selected using validation metrics.
- Classification threshold selected on validation.
- No retraining or tuning on test data.
- Test data read in batches.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from predict import (
    FEATURE_COLUMNS,
    get_model_selection,
    score_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

METADATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "split_metadata.json"
)

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = REPORT_DIR / "test_predictions.csv"
METRICS_PATH = REPORT_DIR / "final_test_metrics.json"

CHUNK_SIZE = 50000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("sentri.final_evaluation")


# ==========================================
# 1. IDENTIFY EXACT MODEL ARTIFACT
# ==========================================

def calculate_sha256(path: Path) -> str:
    """Generate a checksum identifying the saved model file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


# ==========================================
# 2. SCORE COMPLETE HELD-OUT TEST SET
# ==========================================

def score_test_set(
    model_name: str,
    threshold: float,
):
    """Read and score test transactions in batches."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(TEST_PATH)

    labels_parts = []
    scores_parts = []

    total_processed = 0
    temporary_path = REPORT_DIR / "test_predictions.csv.tmp"

    try:
        with temporary_path.open(
            "w", encoding="utf-8", newline=""
        ) as output:

            for batch_number, chunk in enumerate(
                pd.read_csv(
                    TEST_PATH,
                    chunksize=CHUNK_SIZE,
                    usecols=FEATURE_COLUMNS + ["is_fraud"],
                ),
                start=1,
            ):

                labels = chunk["is_fraud"].to_numpy(
                    dtype=np.int8
                )

                if not np.isin(labels, [0, 1]).all():
                    raise ValueError(
                        "Unexpected test fraud labels"
                    )

                scores = score_features(
                    chunk[FEATURE_COLUMNS],
                    model_name,
                )

                predictions = (
                    scores >= threshold
                ).astype(np.int8)

                batch_output = pd.DataFrame({
                    "test_row_number": np.arange(
                        total_processed + 1,
                        total_processed + len(chunk) + 1,
                    ),
                    "type": chunk["type"].to_numpy(),
                    "amount": chunk["amount"].to_numpy(),
                    "actual_fraud": labels,
                    "risk_score": scores,
                    "flagged_for_review": predictions,
                })

                batch_output.to_csv(
                    output,
                    index=False,
                    header=(batch_number == 1),
                )

                labels_parts.append(labels)
                scores_parts.append(scores)

                total_processed += len(chunk)

                logger.info(
                    "Batch %d | Scored %s test rows",
                    batch_number,
                    f"{total_processed:,}",
                )

        if not labels_parts:
            raise RuntimeError("Test dataset was empty")

        temporary_path.replace(PREDICTIONS_PATH)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    y_true = np.concatenate(labels_parts)
    y_scores = np.concatenate(scores_parts)

    return y_true, y_scores


# ==========================================
# 3. COMPUTE FINAL METRICS
# ==========================================

def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
):
    """Compute final metrics using the frozen threshold."""

    if np.unique(y_true).size != 2:
        raise ValueError(
            "Test data must contain both classes"
        )

    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(y_true)

    metrics = {
        "test_transactions": int(total),
        "actual_fraud": int(y_true.sum()),
        "fraud_prevalence": float(
            y_true.mean()
        ),
        "threshold_from_validation": float(threshold),
        "average_precision": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "alerts_generated": int(
            predictions.sum()
        ),
        "alert_rate": float(
            predictions.mean()
        ),
    }

    return metrics


# ==========================================
# 4. CREATE PRECISION-RECALL CURVE
# ==========================================

def plot_precision_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    average_precision: float,
):
    """Visualize the model's precision-recall trade-off."""

    precision, recall, _ = precision_recall_curve(
        y_true,
        scores,
    )

    # Limit plotted points while retaining endpoints
    point_count = min(3000, len(precision))

    positions = np.unique(
        np.linspace(
            0,
            len(precision) - 1,
            point_count,
            dtype=int,
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall[positions],
            y=precision[positions],
            mode="lines",
            name="Selected model",
        )
    )

    baseline = float(y_true.mean())

    fig.add_hline(
        y=baseline,
        line_dash="dash",
        annotation_text="Fraud prevalence baseline",
    )

    fig.update_layout(
        title=(
            "Sentri — Held-Out Test Precision–Recall Curve"
            f" | AP = {average_precision:.4f}"
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )

    fig.write_html(
        FIGURE_DIR / "06_test_precision_recall.html",
        include_plotlyjs=True,
    )


# ==========================================
# 5. CREATE CONFUSION MATRIX
# ==========================================

def plot_confusion_matrix(metrics: dict):
    """Visualize correct and incorrect classifications."""

    matrix = np.array([
        [
            metrics["true_negatives"],
            metrics["false_positives"],
        ],
        [
            metrics["false_negatives"],
            metrics["true_positives"],
        ],
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Not flagged", "Flagged"],
            y=["Legitimate", "Fraudulent"],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text:,}",
            hovertemplate=(
                "Actual: %{y}<br>"
                "Decision: %{x}<br>"
                "Count: %{z:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Sentri — Final Test Confusion Matrix",
        xaxis_title="Model decision",
        yaxis_title="Actual fraud label",
        yaxis=dict(
            autorange="reversed",
        ),
        template="plotly_white",
    )

    fig.write_html(
        FIGURE_DIR / "07_test_confusion_matrix.html",
        include_plotlyjs=True,
    )


# ==========================================
# 6. MODEL COMPARISON FROM VALIDATION
# ==========================================

def plot_validation_comparison(report: dict):
    """Compare models using pre-existing validation AP."""

    model_results = report["model_results"]

    names = list(model_results.keys())

    ap_values = [
        model_results[name]["average_precision"]
        for name in names
    ]

    fig = go.Figure(
        data=go.Bar(
            x=names,
            y=ap_values,
            text=[
                f"{value:.4f}"
                for value in ap_values
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Sentri — Validation Model Comparison",
        xaxis_title="Model",
        yaxis_title="Average Precision",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )

    fig.write_html(
        FIGURE_DIR / "08_validation_model_comparison.html",
        include_plotlyjs=True,
    )


# ==========================================
# 7. MAIN FINAL EVALUATION
# ==========================================

def main():

    logger.info("Starting Sentri final test evaluation")

    model_name, threshold = get_model_selection()

    with (
        REPORT_DIR / "validation_metrics.json"
    ).open("r", encoding="utf-8") as file:
        validation_report = json.load(file)

    model_path = MODEL_DIR / f"{model_name}.joblib"

    model_sha256 = calculate_sha256(model_path)

    logger.info("Selected model: %s", model_name)
    logger.info("Frozen threshold: %.6f", threshold)

    with METADATA_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        split_metadata = json.load(file)

    expected_test_rows = int(
        split_metadata["splits"]["test"]["rows"]
    )

    y_true, scores = score_test_set(
        model_name,
        threshold,
    )

    if len(y_true) != expected_test_rows:
        raise RuntimeError(
            "Test row count does not match split metadata"
        )

    metrics = calculate_metrics(
        y_true,
        scores,
        threshold,
    )

    plot_precision_recall(
        y_true,
        scores,
        metrics["average_precision"],
    )

    plot_confusion_matrix(metrics)

    plot_validation_comparison(
        validation_report
    )

    report = {
        "project": "Sentri",
        "evaluation_type": "Held-out chronological test",
        "model": model_name,
        "model_sha256": model_sha256,
        "model_selection_metric": "validation_average_precision",
        "threshold_selection": "validation_f1_maximum",
        "model_retrained_on_test": False,
        "test_metrics": metrics,
    }

    with METRICS_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    logger.info("FINAL TEST RESULTS")

    for key, value in metrics.items():
        logger.info("%s: %s", key, value)

    logger.info("Saved final metrics and charts")
    logger.info("Sentri final evaluation completed!")


if __name__ == "__main__":
    main()
