import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(page_title="Umba Fraud Detection Dashboard", layout="wide")
st.title("Umba Fraud Detection — Operations Dashboard")

with st.spinner("Loading predictions..."):
    preds = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'predictions.csv'))
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'test.csv'))

    data = test.merge(preds, on='TransactionID', how='left')
    data = data.sort_values('isFraud_prob', ascending=False).reset_index(drop=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions Scored", f"{len(data):,}")
col2.metric("Avg Fraud Probability", f"{data.isFraud_prob.mean():.2%}")
col3.metric("Max Probability", f"{data.isFraud_prob.max():.2%}")
col4.metric("Flagged (p >= 0.5)", f"{(data.isFraud_prob >= 0.5).sum():,}")

st.subheader("Score Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(data['isFraud_prob'], bins=50, color='steelblue', edgecolor='white')
ax.set_xlabel("Fraud Probability")
ax.set_ylabel("Count")
ax.axvline(0.5, color='red', linestyle='--', label='Alarm threshold')
ax.legend()
st.pyplot(fig)

st.subheader("Top Flagged Transactions")
top_k = st.slider("Show top K riskiest transactions", 10, 200, 50)
top = data.head(top_k)[['TransactionID', 'TransactionAmt', 'country', 'currency',
                          'channel', 'card_type', 'isFraud_prob']].copy()
top['isFraud_prob'] = top['isFraud_prob'].map(lambda x: f"{x:.4f}")
top['TransactionAmt'] = top['TransactionAmt'].map(lambda x: f"{x:.2f}")
st.dataframe(top, use_container_width=True)

st.subheader("Operational Metrics at Different Thresholds")
thresholds = np.arange(0.05, 0.95, 0.05)
metrics = []
for t in thresholds:
    flagged = (data['isFraud_prob'] >= t).sum()
    metrics.append({'Threshold': f"{t:.0%}", 'Flagged Count': flagged,
                    'Flagged %': f"{flagged/len(data):.2%}"})
metrics_df = pd.DataFrame(metrics)
st.dataframe(metrics_df, use_container_width=True)

st.subheader("Fraud by Channel")
channel_stats = data.groupby('channel').agg(
    total=('TransactionID', 'count'),
    mean_prob=('isFraud_prob', 'mean'),
    high_risk=('isFraud_prob', lambda x: (x >= 0.5).sum())
).reset_index()
st.dataframe(channel_stats, use_container_width=True)

st.subheader("Fraud by Country")
country_stats = data.groupby('country').agg(
    total=('TransactionID', 'count'),
    mean_prob=('isFraud_prob', 'mean'),
    high_risk=('isFraud_prob', lambda x: (x >= 0.5).sum())
).reset_index()
st.dataframe(country_stats, use_container_width=True)

st.markdown("---")
st.caption("Umba Fraud Detection v1 — Operations Dashboard")
