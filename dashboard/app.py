1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
"""Sentri — local fraud intelligence dashboard (Step 13)."""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'dashboard' else Path(__file__).parent
sys.path.insert(0, str(ROOT / 'src'))

from database import engine  # noqa: E402
from predict import predict_transaction  # noqa: E402

REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'

st.set_page_config(page_title='Sentri | Fraud Intelligence', page_icon='🛡️', layout='wide')
st.markdown('''
<style>
.stApp {background-color:#0B1420; color:#E7EDF4;}
[data-testid="stSidebar"] {background-color:#101C2B;}
[data-testid="stMetric"] {background:#152337; border:1px solid #29394B;
  padding:14px; border-radius:12px;}
h1,h2,h3 {letter-spacing:-.025em;}
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=600, show_spinner='Loading transaction analytics...')
def query(sql: str) -> pd.DataFrame:
    """Run read-only SQL and cache the small aggregated result for 10 minutes."""
    with engine.connect() as connection:
        return pd.read_sql_query(text(sql), connection)

@st.cache_data(ttl=600)
def load_json(path_string: str) -> dict:
    with open(path_string, encoding='utf-8') as file:
        return json.load(file)

@st.cache_data(ttl=600, show_spinner='Reading saved fraud alerts...')
def load_alerts(max_rows: int = 1000) -> pd.DataFrame:
    """Keep up to max_rows actual model alerts without loading the entire CSV."""
    path = REPORTS / 'test_predictions.csv'
    if not path.exists():
        return pd.DataFrame()
    parts = []
    count = 0
    for chunk in pd.read_csv(path, chunksize=50000):
        flagged = chunk.loc[chunk['flagged_for_review'] == 1]
        if len(flagged):
            selected = flagged.head(max_rows - count)
            parts.append(selected)
            count += len(selected)
        if count >= max_rows:
            break
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def page_header(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)


def overview():
    page_header('Transaction Overview', 'PaySim synthetic transaction analytics · PostgreSQL')
    totals = query('''
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE is_fraud=1) AS fraud,
               COALESCE(SUM(amount),0) AS volume,
               COALESCE(SUM(amount) FILTER (WHERE is_fraud=1),0) AS fraud_volume
        FROM raw_transactions
    ''').iloc[0]
    total, fraud = int(totals['total']), int(totals['fraud'])
    a,b,c,d = st.columns(4)
    a.metric('Transactions', f'{total:,}')
    b.metric('Labeled fraud', f'{fraud:,}')
    c.metric('Fraud rate', f'{fraud / total * 100:.3f}%' if total else 'N/A')
    d.metric('Fraud amount (dataset units)', f"{float(totals['fraud_volume']):,.0f}")
    st.caption('Amounts are synthetic dataset units, not verified INR or monetary loss.')

    type_df = query('''
        SELECT type, COUNT(*) AS total,
               COUNT(*) FILTER (WHERE is_fraud=1) AS fraud
        FROM raw_transactions GROUP BY type ORDER BY type
    ''')
    type_df['fraud_rate_pct'] = 100 * type_df['fraud'] / type_df['total']
    timeline = query('''
        SELECT step, COUNT(*) AS transactions,
               COUNT(*) FILTER (WHERE is_fraud=1) AS fraud
        FROM raw_transactions GROUP BY step ORDER BY step
    ''')
    left, right = st.columns(2)