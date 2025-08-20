# Loan Default Risk & Expected Loss Simulator (Finance ML)

A compact, portfolio-ready credit risk project. We build a model to estimate probability of default (PD) for loans, compute Expected Loss (EL = PD × LGD × EAD), and run macro stress scenarios.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make quickstart    # generate synthetic loans -> features -> train -> evaluate
make app           # launch the dashboard
```