
-- SENTRI: Add source tracking for idempotent ingestion

ALTER TABLE raw_transactions
ADD COLUMN IF NOT EXISTS source_file TEXT;

ALTER TABLE raw_transactions
ADD COLUMN IF NOT EXISTS source_row_number BIGINT;

-- Backfill the original 1,000 imported transactions.
-- Only do this if all existing records are still untracked.

DO $$
DECLARE
    untracked_count BIGINT;
BEGIN
    SELECT COUNT(*)
    INTO untracked_count
    FROM raw_transactions
    WHERE source_row_number IS NULL;

    IF untracked_count > 0 THEN

        IF untracked_count <> 1000 THEN
            RAISE EXCEPTION
                'Expected 1000 original rows, found %',
                untracked_count;
        END IF;

        WITH numbered AS (
            SELECT
                transaction_id,
                ROW_NUMBER() OVER (
                    ORDER BY transaction_id
                ) AS row_num
            FROM raw_transactions
            WHERE source_row_number IS NULL
        )
        UPDATE raw_transactions AS t
        SET
            source_file = 'paysim.csv',
            source_row_number = n.row_num
        FROM numbered AS n
        WHERE t.transaction_id = n.transaction_id;

    END IF;
END $$;

-- Ensure future records have source identifiers.
ALTER TABLE raw_transactions
ALTER COLUMN source_file SET NOT NULL;

ALTER TABLE raw_transactions
ALTER COLUMN source_row_number SET NOT NULL;

-- One source row may only appear once.
CREATE UNIQUE INDEX IF NOT EXISTS
    ux_raw_transactions_source
ON raw_transactions (
    source_file,
    source_row_number
);
