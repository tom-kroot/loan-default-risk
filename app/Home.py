import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

st.set_page_config(page_title="Credit Risk — Expected Loss Simulator", layout="wide")
st.title("💳 Loan Default Risk & Expected Loss — Demo")

proc = Path("data/processed")
mfile = proc/"metrics.json"
pfile = proc/"predictions.csv"
tfile = proc/"test.csv"
model_file = proc/"model.joblib"

if mfile.exists():
    m = pd.read_json(mfile, typ="series")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
    c2.metric("PR-AUC",  f"{m.get('pr_auc', float('nan')):.3f}")
    c3.metric("Brier",   f"{m.get('brier', float('nan')):.3f}")
else:
    st.info("Run `make quickstart` to generate data, train, and evaluate.")

st.header("Portfolio — Expected Loss & Income")
if mfile.exists():
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Loss (EL)", f"${m.get('expected_loss', 0):,.0f}")
    c2.metric("Expected Interest", f"${m.get('expected_interest_income', 0):,.0f}")
    c3.metric("Expected Net", f"${m.get('expected_net', 0):,.0f}")

st.header("Macro Stress Test")
if tfile.exists() and model_file.exists():
    test = pd.read_csv(tfile, parse_dates=["issue_date"])
    X = test[["fico","dti_ratio","log_income","term_months","log_loan","apr","unemp_rate","month"]].astype(float)
    model = joblib.load(model_file)
    base_pd = model.predict_proba(X)[:,1]

    shock = st.slider("Unemployment Shock (Δ percentage points)", -2.0, 5.0, 0.0, 0.5)
    beta = 0.5
    logit = np.log(np.clip(base_pd,1e-6,1-1e-6)/np.clip(1-base_pd,1e-6,1))
    stressed_pd = 1/(1+np.exp(-(logit + beta*shock)))

    ead = test["ead"].to_numpy()
    lgd = test["lgd"].to_numpy()
    apr = test["apr"].to_numpy()
    exp_loss = np.sum(stressed_pd * lgd * ead)
    exp_int = np.sum((1 - stressed_pd) * apr * ead)
    exp_net = exp_int - exp_loss

    c1, c2, c3 = st.columns(3)
    c1.metric("Stressed EL", f"${exp_loss:,.0f}")
    c2.metric("Stressed Interest", f"${exp_int:,.0f}")
    c3.metric("Stressed Net", f"${exp_net:,.0f}")

st.header("Loans — Top Risk (Test Set)")
if pfile.exists():
    preds = pd.read_csv(pfile).sort_values("pd_hat", ascending=False).head(50)
    st.dataframe(preds)
else:
    st.warning("No predictions yet.")

st.caption("Synthetic data; demo only. Not investment advice.")