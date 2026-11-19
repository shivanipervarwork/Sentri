
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert

from database import engine
from validate import validate_chunk

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
LOG_DIR = PROJECT_ROOT / "logs"

BATCH_SIZE = 50000
INSERT_SIZE = 2000

LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingestion.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("sentri.ingestion")

# Load the existing PostgreSQL table definition
metadata = MetaData()

transactions = Table(
    "raw_transactions",
    metadata,
    autoload_with=engine,
)



def insert_batch(df: pd.DataFrame) -> int:
    """Insert transactions and count the actual inserted rows."""

    if df.empty:
        return 0

    inserted_count = 0

    records = (
        df.astype(object)
        .where(pd.notna(df), None)
        .to_dict(orient="records")
    )

    with engine.begin() as connection:

        for start in range(0, len(records), INSERT_SIZE):
            group = records[start:start + INSERT_SIZE]

            statement = (
                insert(transactions)
                .on_conflict_do_nothing(
                    index_elements=[
                        "source_file",
                        "source_row_number",
                    ]
                )
                .returning(transactions.c.transaction_id)
            )

            result = connection.execute(statement, group)

            inserted_count += len(result.fetchall())

    return inserted_count





def run_ingestion():
    """Read, validate and ingest the full PaySim dataset."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    total_read = 0
    total_inserted = 0
    total_rejected = 0
    total_duplicates = 0
    batch_number = 0

    logger.info("Starting Sentri ingestion")
    logger.info("Dataset: %s", DATA_PATH.name)

    for chunk in pd.read_csv(
    DATA_PATH,
    chunksize=BATCH_SIZE,
    ):
        batch_number += 1
        chunk_size = len(chunk)

        # Track each row's position in the source CSV
        chunk["source_file"] = DATA_PATH.name
        chunk["source_row_number"] = range(
            total_read + 1,
            total_read + chunk_size + 1,
        )

        clean, rejected, reasons = validate_chunk(chunk)

        inserted = insert_batch(clean)
        duplicates = len(clean) - inserted

        total_read += chunk_size
        total_inserted += inserted
        total_rejected += rejected
        total_duplicates += duplicates

        logger.info(
            "Batch %d | Read: %d | Inserted: %d | "
            "Rejected: %d | Already present: %d",
            batch_number,
            chunk_size,
            inserted,
            rejected,
            duplicates,
        )

        if rejected:
            logger.warning(
                "Batch %d validation failures: %s",
                batch_number,
                reasons,
            )

    logger.info("INGESTION COMPLETED")
    logger.info("Total rows read: %d", total_read)
    logger.info("New rows inserted: %d", total_inserted)
    logger.info("Rejected rows: %d", total_rejected)
    logger.info(
        "Previously imported rows: %d",
        total_duplicates,
    )


if __name__ == "__main__":
    try:
        run_ingestion()
    except Exception:
        logger.exception("Sentri ingestion failed")
        raise
