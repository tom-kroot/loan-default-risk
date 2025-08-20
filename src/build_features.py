import pandas as pd, numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

def build():
    df = pd.read_csv(RAW/"loans.csv", parse_dates=["issue_date"])
    df["month"] = df["issue_date"].dt.month
    df["year"] = df["issue_date"].dt.year
    df["log_income"] = np.log1p(df["income"])
    df["log_loan"] = np.log1p(df["loan_amount"])
    df["dti_ratio"] = df["dti"]/100.0
    feat_cols = ["fico","dti_ratio","log_income","term_months","log_loan","apr","unemp_rate","month"]
    target = "default_12m"
    keep = ["issue_date","loan_id",target] + feat_cols + ["ead","lgd"]
    df = df[keep].dropna().reset_index(drop=True)

    max_date = df["issue_date"].max()
    test_cut = max_date - pd.Timedelta(days=60)
    valid_cut = test_cut - pd.Timedelta(days=60)
    train = df[df["issue_date"] <= valid_cut]
    valid = df[(df["issue_date"] > valid_cut) & (df["issue_date"] <= test_cut)]
    test  = df[df["issue_date"] > test_cut]

    for name, split in [("train",train),("valid",valid),("test",test)]:
        split.to_csv(PROC/f"{name}.csv", index=False)
    print("Built features → data/processed/{train,valid,test}.csv")

if __name__ == "__main__":
    build()