# 💳 Loan Default Risk & Expected Loss Simulator (Finance ML)

Portfolio-ready **credit risk** project. Trains a model to estimate **Probability of Default (PD)** for synthetic loans, then computes **Expected Loss (EL = PD × LGD × EAD)** and shows **portfolio impact** under macro stress (unemployment shock) in a Streamlit app.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Build](https://github.com/tom-kroot/loan-default-risk/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## 🔎 What this demonstrates
- Credit risk framing: **PD / LGD / EAD** and **Expected Loss**
- **Imbalanced classification** with metrics that matter (PR-AUC, ROC-AUC, Brier/calibration)
- **Macro stress** slider (unemployment shock) to see EL & Net move
- Clean pipeline + **Streamlit** dashboard (recruiter-friendly)

## 🚀 Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make quickstart    # data → features → train → evaluate
make app           # open http://localhost:8501


📂 Structure
loan_default_risk_project/
├─ app/Home.py              # Streamlit dashboard + stress test
├─ src/                     # data gen, features, training, evaluation
├─ data/                    # raw/processed (gitignored)
├─ tests/                   # basic tests
├─ Makefile                 # quickstart/app
├─ requirements.txt
└─ .github/workflows/ci.yml # CI stub



