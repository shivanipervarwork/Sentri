
-- SENTRI: Database Schema
-- Stores raw financial transaction records from PaySim.

CREATE TABLE IF NOT EXISTS raw_transactions (
    transaction_id BIGSERIAL PRIMARY KEY,

    step INTEGER NOT NULL,
    type VARCHAR(20) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,

    name_orig VARCHAR(30),
    oldbalance_org DOUBLE PRECISION,
    newbalance_orig DOUBLE PRECISION,

    name_dest VARCHAR(30),
    oldbalance_dest DOUBLE PRECISION,
    newbalance_dest DOUBLE PRECISION,

    is_fraud SMALLINT NOT NULL,
    is_flagged_fraud SMALLINT,

    ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT valid_amount CHECK (amount >= 0),
    CONSTRAINT valid_step CHECK (step >= 1),
    CONSTRAINT valid_fraud_label CHECK (is_fraud IN (0, 1)),
    CONSTRAINT valid_flag_label CHECK (is_flagged_fraud IN (0, 1))
);
