
-- ============================================
-- SENTRI: TRANSACTION & FRAUD ANALYTICS
-- ============================================

-- QUERY 1: Overall transaction statistics

SELECT
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (
        WHERE is_fraud = 1
    ) AS fraudulent_transactions,
    COUNT(*) FILTER (
        WHERE is_fraud = 0
    ) AS legitimate_transactions,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE is_fraud = 1
        ) / NULLIF(COUNT(*), 0),
        4
    ) AS fraud_percentage
FROM raw_transactions;


-- QUERY 2: Fraud by transaction type

SELECT
    type,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (
        WHERE is_fraud = 1
    ) AS fraud_count,
    ROUND(
        100.0 * COUNT(*) FILTER (
            WHERE is_fraud = 1
        ) / NULLIF(COUNT(*), 0),
        4
    ) AS fraud_rate
FROM raw_transactions
GROUP BY type
ORDER BY fraud_rate DESC;


-- QUERY 3: Transaction amount statistics

SELECT
    type,
    ROUND(AVG(amount)::numeric, 2)
        AS average_amount,
    ROUND(MAX(amount)::numeric, 2)
        AS maximum_amount,
    ROUND(SUM(amount)::numeric, 2)
        AS total_amount
FROM raw_transactions
GROUP BY type
ORDER BY total_amount DESC;


-- QUERY 4: Financial value of fraudulent transactions

SELECT
    COUNT(*) AS fraud_transactions,
    ROUND(SUM(amount)::numeric, 2)
        AS total_fraud_amount,
    ROUND(AVG(amount)::numeric, 2)
        AS average_fraud_amount,
    ROUND(MAX(amount)::numeric, 2)
        AS maximum_fraud_amount
FROM raw_transactions
WHERE is_fraud = 1;


-- QUERY 5: Fraud activity over simulation time

SELECT
    step,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (
        WHERE is_fraud = 1
    ) AS fraud_transactions
FROM raw_transactions
GROUP BY step
ORDER BY step;


-- QUERY 6: Existing rule-based flags

SELECT
    is_flagged_fraud,
    is_fraud,
    COUNT(*) AS transaction_count
FROM raw_transactions
GROUP BY is_flagged_fraud, is_fraud
ORDER BY is_flagged_fraud, is_fraud;


-- QUERY 7: Top sender accounts by activity

SELECT
    name_orig,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount)::numeric, 2)
        AS total_amount,
    COUNT(*) FILTER (
        WHERE is_fraud = 1
    ) AS fraud_count
FROM raw_transactions
GROUP BY name_orig
ORDER BY transaction_count DESC
LIMIT 10;
