import pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

PROC = Path("data/processed")

def business_metrics(df, p):
    el = float(np.sum(p * df["lgd"].to_numpy() * df["ead"].to_numpy()))
    interest_income = float(np.sum((1 - p) * df["ead"].to_numpy() * df["apr"].to_numpy()))
    net = interest_income - el
    return {"expected_loss": el, "expected_interest_income": interest_income, "expected_net": net}

def main():
    test = pd.read_csv(PROC/"test.csv", parse_dates=["issue_date"])
    X = test[["fico","dti_ratio","log_income","term_months","log_loan","apr","unemp_rate","month"]].astype(float).to_numpy()
    y = test["default_12m"].astype(int).to_numpy()
    model = joblib.load(PROC/"model.joblib")
    p = model.predict_proba(X)[:,1]

    roc = float(roc_auc_score(y, p))
    pr = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))

    out = test[["issue_date","loan_id","ead","lgd","apr"]].copy()
    out["y_true"] = y
    out["pd_hat"] = p
    out.to_csv(PROC/"predictions.csv", index=False)

    port = business_metrics(test, p)

    pd.Series({"roc_auc": roc, "pr_auc": pr, "brier": brier, **port}).to_json(PROC/"metrics.json")
    print(f"Test — ROC-AUC: {roc:.3f}, PR-AUC: {pr:.3f}, Brier: {brier:.3f}")
    print(f"Expected Net: {port['expected_net']:.2f}")

if __name__ == "__main__":
    main()