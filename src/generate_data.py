import numpy as np, pandas as pd
from pathlib import Path

RAW = Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)

def simulate(n_loans=5000, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=540, freq="D")
    issue_dates = rng.choice(dates, size=n_loans)

    fico = rng.normal(690, 40, n_loans).clip(500, 850)
    dti = rng.uniform(5, 45, n_loans)
    income = np.exp(rng.normal(np.log(120000), 0.6, n_loans))
    term_months = rng.choice([24,36,48,60], p=[0.2,0.4,0.25,0.15], size=n_loans)
    loan_amount = rng.gamma(shape=2.5, scale=8000, size=n_loans).clip(3000, 150000)
    rate = (0.06 + 0.12*(700 - fico).clip(0)/200) + 0.005*(dti/45)
    rate = rate.clip(0.05, 0.24)

    months = pd.to_datetime(pd.Series(issue_dates)).dt.to_period("M").astype(str)
    uniq_months = sorted(pd.unique(months))
    base_unemp = 0.035
    month_idx = {m:i for i,m in enumerate(uniq_months)}
    unemp_map = {m: base_unemp + 0.002*np.sin(i/3) for m,i in month_idx.items()}
    unemp = np.array([unemp_map[m] for m in months])

    ead = loan_amount
    lgd = rng.uniform(0.2, 0.7, n_loans)

    z = (-6.0 + 0.04*(rate*100) + 0.03*(dti) - 0.01*(fico-700) + 5.0*(unemp)
         - 0.000003*(income) + 0.000004*(loan_amount))
    p_default = 1/(1+np.exp(-z))
    default = (rng.uniform(0,1,n_loans) < p_default).astype(int)

    df = pd.DataFrame({
        "loan_id": np.arange(1, n_loans+1),
        "issue_date": issue_dates,
        "fico": np.round(fico,0),
        "dti": dti,
        "income": income,
        "term_months": term_months,
        "loan_amount": loan_amount,
        "apr": rate,
        "unemp_rate": unemp,
        "ead": ead,
        "lgd": lgd,
        "default_12m": default
    }).sort_values("issue_date").reset_index(drop=True)

    df.to_csv(RAW/"loans.csv", index=False)
    print("Wrote data/raw/loans.csv")

if __name__ == "__main__":
    simulate()