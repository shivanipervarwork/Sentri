
"""
SENTRI — Exploratory Data Analysis

Reads aggregated transaction statistics from PostgreSQL
and produces interactive Plotly visualizations.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import text

from database import engine


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. DATABASE QUERY HELPER
# ==========================================

def run_query(sql: str) -> pd.DataFrame:
    """Execute SQL and return results as a DataFrame."""

    with engine.connect() as connection:
        return pd.read_sql_query(
            text(sql),
            connection,
        )


# ==========================================
# 3. SAVE CHART HELPER
# ==========================================

def save_chart(fig, filename: str):
    """Save an interactive Plotly chart as HTML."""

    output_path = OUTPUT_DIR / filename

    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
    )

    print(f"Saved chart: {output_path.name}")


# ==========================================
# 4. FRAUD DISTRIBUTION
# ==========================================

def plot_fraud_distribution():

    sql = """
    SELECT
        is_fraud,
        COUNT(*) AS transaction_count
    FROM raw_transactions
    GROUP BY is_fraud
    ORDER BY is_fraud;
    """

    df = run_query(sql)

    df["category"] = df["is_fraud"].map({
        0: "Legitimate",
        1: "Fraudulent",
    })

    df["transaction_count"] = pd.to_numeric(
        df["transaction_count"]
    )

    fig = px.bar(
        df,
        x="category",
        y="transaction_count",
        color="category",
        text="transaction_count",
        title="Fraud vs Legitimate Transactions",
        labels={
            "category": "Transaction classification",
            "transaction_count": "Number of transactions",
        },
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        yaxis_type="log",
    )

    save_chart(fig, "01_fraud_distribution.html")


# ==========================================
# 5. FRAUD RATE BY TRANSACTION TYPE
# ==========================================

def plot_fraud_by_type():

    sql = """
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
    """

    df = run_query(sql)

    df["fraud_rate"] = pd.to_numeric(
        df["fraud_rate"]
    )

    fig = px.bar(
        df,
        x="type",
        y="fraud_rate",
        color="type",
        text="fraud_rate",
        title="Fraud Rate by Transaction Type",
        labels={
            "type": "Transaction type",
            "fraud_rate": "Fraud rate (%)",
        },
        hover_data=[
            "total_transactions",
            "fraud_count",
        ],
    )

    fig.update_traces(
        texttemplate="%{text:.3f}%",
        textposition="outside",
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    save_chart(fig, "02_fraud_by_type.html")


# ==========================================
# 6. TRANSACTION ACTIVITY OVER TIME
# ==========================================

def plot_transaction_activity():

    sql = """
    SELECT
        step,
        COUNT(*) AS total_transactions,
        COUNT(*) FILTER (
            WHERE is_fraud = 1
        ) AS fraudulent_transactions
    FROM raw_transactions
    GROUP BY step
    ORDER BY step;
    """

    df = run_query(sql)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["total_transactions"],
            mode="lines",
            name="All transactions",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["fraudulent_transactions"],
            mode="lines",
            name="Fraudulent transactions",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Transaction Activity Over Simulation Time",
        xaxis_title="Simulation hour (step)",
        yaxis=dict(
            title="All transactions",
        ),
        yaxis2=dict(
            title="Fraudulent transactions",
            overlaying="y",
            side="right",
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.12,
        ),
    )

    save_chart(fig, "03_transaction_activity.html")


# ==========================================
# 7. FRAUDULENT AMOUNT DISTRIBUTION
# ==========================================

def plot_fraud_amount_distribution():

    sql = """
    SELECT
        CASE
            WHEN amount = 0 THEN '0'
            WHEN amount <= 1000 THEN '0-1K'
            WHEN amount <= 10000 THEN '1K-10K'
            WHEN amount <= 100000 THEN '10K-100K'
            WHEN amount <= 1000000 THEN '100K-1M'
            ELSE 'Above 1M'
        END AS amount_range,
        COUNT(*) AS transaction_count
    FROM raw_transactions
    WHERE is_fraud = 1
    GROUP BY 1;
    """

    df = run_query(sql)

    category_order = [
        "0",
        "0-1K",
        "1K-10K",
        "10K-100K",
        "100K-1M",
        "Above 1M",
    ]

    fig = px.bar(
        df,
        x="amount_range",
        y="transaction_count",
        category_orders={
            "amount_range": category_order,
        },
        title="Distribution of Fraudulent Transaction Amounts",
        labels={
            "amount_range": "Amount range (dataset units)",
            "transaction_count": "Fraudulent transactions",
        },
        text="transaction_count",
    )

    fig.update_traces(
        textposition="outside",
    )

    fig.update_layout(
        template="plotly_white",
    )

    save_chart(fig, "04_fraud_amounts.html")


# ==========================================
# 8. EXISTING RULE-BASED FRAUD FLAGS
# ==========================================

def plot_existing_flags():

    sql = """
    SELECT
        is_flagged_fraud,
        is_fraud,
        COUNT(*) AS transaction_count
    FROM raw_transactions
    GROUP BY is_flagged_fraud, is_fraud;
    """

    df = run_query(sql)

    matrix = (
        df.pivot(
            index="is_fraud",
            columns="is_flagged_fraud",
            values="transaction_count",
        )
        .reindex(index=[0, 1], columns=[0, 1])
        .fillna(0)
        .astype(int)
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=[
                "Not flagged",
                "Flagged",
            ],
            y=[
                "Legitimate",
                "Fraudulent",
            ],
            colorscale="Blues",
            text=matrix.values,
            texttemplate="%{text:,}",
            hovertemplate=(
                "Actual: %{y}<br>"
                "Existing rule: %{x}<br>"
                "Transactions: %{z:,}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Existing PaySim Rule vs Actual Fraud Labels",
        xaxis_title="Existing rule-based flag",
        yaxis_title="Actual dataset label",
        template="plotly_white",
    )

    save_chart(fig, "05_existing_flags.html")


# ==========================================
# 9. MAIN EXECUTION
# ==========================================

def main():

    print("=" * 45)
    print("SENTRI — EXPLORATORY DATA ANALYSIS")
    print("=" * 45)

    total = run_query(
        "SELECT COUNT(*) AS total FROM raw_transactions;"
    )

    row_count = int(total.loc[0, "total"])

    print(f"Transactions in PostgreSQL: {row_count:,}")

    if row_count == 0:
        raise RuntimeError(
            "The database has no transactions to analyze."
        )

    plot_fraud_distribution()
    plot_fraud_by_type()
    plot_transaction_activity()
    plot_fraud_amount_distribution()
    plot_existing_flags()

    print("\nEDA completed successfully!")
    print(f"Charts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
