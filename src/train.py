import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score
import joblib

PROC = Path("data/processed")

def load(name):
    df = pd.read_csv(PROC/f"{name}.csv", parse_dates=["issue_date"])
    y = df["default_12m"].astype(int).to_numpy()
    X = df[["fico","dti_ratio","log_income","term_months","log_loan","apr","unemp_rate","month"]].astype(float).to_numpy()
    return X, y, df

def main():
    X_train, y_train, _ = load("train")
    X_valid, y_valid, _ = load("valid")

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    p_lr = lr.predict_proba(X_valid)[:,1]
    pr_lr = average_precision_score(y_valid, p_lr)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    p_gb = gb.predict_proba(X_valid)[:,1]
    pr_gb = average_precision_score(y_valid, p_gb)

    best = ("lr", lr, pr_lr) if pr_lr >= pr_gb else ("gb", gb, pr_gb)
    model = lr if best[0]=="lr" else gb

    joblib.dump(model, PROC/"model.joblib")
    pd.Series({"best_model": best[0], "pr_auc_valid": float(best[1])}).to_json(PROC/"train_summary.json")
    print(f"Trained. Best model: {best[0]} (PR-AUC={best[1]:.3f})")

if __name__ == "__main__":
    main()