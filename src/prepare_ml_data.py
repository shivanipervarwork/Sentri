
"""
SENTRI — ML Dataset Preparation

1. Examine transaction counts by simulation step.
2. Determine chronological split boundaries.
3. Verify fraud examples exist in every partition.
4. Extract PostgreSQL records in batches.
5. Engineer features.
6. Export train, validation, and test CSV files.
7. Save split metadata.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from database import engine
from features import build_features


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 50000

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("sentri.ml_preparation")


# ==========================================
# STEP 1: GET SOURCE COUNTS BY SIMULATION STEP
# ==========================================

def get_step_statistics() -> pd.DataFrame:
    """Count transactions and fraud cases per simulation step."""

    query = text("""
        SELECT
            step,
            COUNT(*) AS total_transactions,
            COUNT(*) FILTER (
                WHERE is_fraud = 1
            ) AS fraud_transactions
        FROM raw_transactions
        GROUP BY step
        ORDER BY step
    """)

    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)

    if df.empty:
        raise ValueError(
            "No transactions found in PostgreSQL."
        )

    return df


# ==========================================
# STEP 2: CHOOSE CHRONOLOGICAL BOUNDARIES
# ==========================================

def choose_split_boundaries(stats: pd.DataFrame):
    """Find simulation steps nearest 70% and 85% of records."""

    stats = stats.sort_values("step").copy()

    if len(stats) < 3:
        raise ValueError(
            "At least three simulation steps are required."
        )

    total_rows = int(stats["total_transactions"].sum())

    cumulative = stats["total_transactions"].cumsum()

    train_target = total_rows * TRAIN_FRACTION

    validation_target = total_rows * (
        TRAIN_FRACTION + VALIDATION_FRACTION
    )

    train_position = cumulative.searchsorted(
        train_target, side="left"
    )

    validation_position = cumulative.searchsorted(
        validation_target, side="left"
    )

    train_position = int(train_position)
    validation_position = int(validation_position)

    if train_position >= validation_position:
        raise ValueError(
            "Chronological boundaries overlap."
        )

    if validation_position >= len(stats) - 1:
        raise ValueError(
            "No simulation steps remain for the test set."
        )

    train_end = int(
        stats.iloc[train_position]["step"]
    )

    validation_end = int(
        stats.iloc[validation_position]["step"]
    )

    return train_end, validation_end


# ==========================================
# STEP 3: CHECK CLASS DISTRIBUTIONS
# ==========================================

def describe_splits(
    stats: pd.DataFrame,
    train_end: int,
    validation_end: int,
):
    """Calculate row and fraud counts for each partition."""

    conditions = {
        "train": stats["step"] <= train_end,

        "validation": (
            (stats["step"] > train_end)
            & (stats["step"] <= validation_end)
        ),

        "test": stats["step"] > validation_end,
    }

    results = {}

    for split_name, mask in conditions.items():
        subset = stats.loc[mask]

        rows = int(
            subset["total_transactions"].sum()
        )

        fraud = int(
            subset["fraud_transactions"].sum()
        )

        if rows == 0:
            raise ValueError(
                f"{split_name} is empty."
            )

        if fraud == 0:
            raise ValueError(
                f"{split_name} has no fraud examples."
            )

        results[split_name] = {
            "rows": rows,
            "fraud": fraud,
            "legitimate": rows - fraud,
            "fraud_rate_percent": round(
                fraud / rows * 100, 4
            ),
            "min_step": int(subset["step"].min()),
            "max_step": int(subset["step"].max()),
        }

    return results


# ==========================================
# STEP 4: EXPORT ONE PARTITION
# ==========================================

def export_split(
    split_name: str,
    min_step: int,
    max_step: int,
):
    """Read a time range in chunks and save ML features."""

    output_path = OUTPUT_DIR / f"{split_name}.csv"
    temporary_path = OUTPUT_DIR / f"{split_name}.csv.tmp"

    query = text("""
        SELECT
            step,
            type,
            amount,
            is_fraud
        FROM raw_transactions
        WHERE step >= :min_step
          AND step <= :max_step
    """)

    total_exported = 0
    header_written = False

    try:
        with engine.connect().execution_options(
            stream_results=True
        ) as connection:

            chunks = pd.read_sql_query(
                query,
                connection,
                params={
                    "min_step": min_step,
                    "max_step": max_step,
                },
                chunksize=CHUNK_SIZE,
            )

            with temporary_path.open(
                "w", encoding="utf-8", newline=""
            ) as file:

                for chunk_number, chunk in enumerate(
                    chunks, start=1
                ):

                    features = build_features(chunk)

                    features.to_csv(
                        file,
                        index=False,
                        header=not header_written,
                    )

                    header_written = True

                    total_exported += len(features)

                    logger.info(
                        "%s | Chunk %d | Exported %d rows",
                        split_name,
                        chunk_number,
                        total_exported,
                    )

        if total_exported == 0:
            raise ValueError(
                f"No rows exported for {split_name}"
            )

        # Replace the final output only after success
        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return total_exported


# ==========================================
# STEP 5: MAIN PIPELINE
# ==========================================

def main():

    logger.info("Starting Sentri ML dataset preparation")

    stats = get_step_statistics()

    total_rows = int(
        stats["total_transactions"].sum()
    )

    logger.info(
        "Total transactions found: %s",
        f"{total_rows:,}",
    )

    train_end, validation_end = (
        choose_split_boundaries(stats)
    )

    logger.info("Training ends at step %d", train_end)

    logger.info(
        "Validation ends at step %d",
        validation_end,
    )

    split_stats = describe_splits(
        stats,
        train_end,
        validation_end,
    )

    for split_name, details in split_stats.items():

        logger.info(
            "%s | Rows: %d | Fraud: %d | Rate: %.4f%%",
            split_name,
            details["rows"],
            details["fraud"],
            details["fraud_rate_percent"],
        )

    # Export the three non-overlapping periods
    for split_name, details in split_stats.items():

        exported = export_split(
            split_name,
            details["min_step"],
            details["max_step"],
        )

        if exported != details["rows"]:
            raise RuntimeError(
                f"{split_name}: expected {details['rows']} "
                f"rows, exported {exported}"
            )

    # Record the exact split and dataset information
    metadata = {
        "project": "Sentri",
        "dataset": "PaySim synthetic transactions",
        "total_rows": total_rows,
        "train_end_step": train_end,
        "validation_end_step": validation_end,
        "split_method": "Chronological by simulation step",
        "feature_columns": [
            "amount",
            "log_amount",
            "type",
            "hour_sin",
            "hour_cos",
        ],
        "target_column": "is_fraud",
        "splits": split_stats,
    }

    metadata_path = OUTPUT_DIR / "split_metadata.json"

    with metadata_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    logger.info("ML dataset preparation completed!")
    logger.info(
        "Files saved to: %s",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
