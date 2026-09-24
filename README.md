# SAGE Price Model

> Performance-weighted price forecasting research prototype for quantitative investment research.

## Overview

SAGE Price Model is a research-oriented project that explores how multiple price forecasts can be combined into a single investment signal. Instead of relying on one prediction model, the framework is designed to assign larger weights to forecasts that show stronger validation performance and then aggregate them into an expected-return signal.

The project is intended as a **decision-support research model**, not as a standalone trading system. The next step is to evaluate whether the signal remains useful under out-of-sample testing, different market regimes, transaction costs, and portfolio risk constraints.

## Research Question

> Can multiple forecasting outputs be combined according to validation performance to produce a more robust market signal than relying on a single prediction output?

## Core Idea

```text
Market Data
    ↓
Data Preprocessing
    ↓
Multiple Forecast Outputs
    ↓
Validation Performance Measurement
    ↓
Performance-based Weighting
    ↓
Aggregated Expected-Return Signal
    ↓
Out-of-Sample / Portfolio Evaluation
```

The key principle is that a model should not receive a large weight simply because it fits historical data well. Its weight should be based on a clearly separated validation period, followed by evaluation on unseen data.

## Why This Matters

Financial time series are noisy and market regimes change. A single prediction model can easily overfit a specific sample. This project therefore focuses on three issues:

- **Model combination** rather than dependence on one forecast
- **Validation-based weighting** rather than in-sample performance
- **Risk-aware use of predictions** rather than treating forecasts as certain outcomes

## Evaluation Framework

The model should be evaluated with both forecasting and investment metrics.

### Forecast evaluation

- MAE / RMSE for continuous forecasts
- Directional accuracy where appropriate
- Stability across validation and test periods

### Investment evaluation

- Cumulative return
- CAGR
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Turnover and transaction-cost sensitivity

## Validation Rules

To reduce the risk of overstating backtest performance, the project follows these principles:

1. Train, validation, and test periods must remain separated.
2. Performance weights must be determined without using future test data.
3. Signals must be shifted when necessary so that information available at time `t` is used only for decisions after time `t`.
4. Trading costs and turnover should be included before interpreting investment performance.
5. Results should be checked across different market regimes rather than on one favorable interval.

## Connection to RoboBridge

A related asset-management concept, **RoboBridge**, treats the performance-weighted prediction model as a possible future extension rather than an already validated production component. The intended use is to convert model outputs into a limited portfolio-adjustment signal while keeping user suitability and risk limits as constraints.

This distinction is important: the forecasting model is a research component whose performance must be validated before it is used in an investment service.

## Current Status

Research prototype / ongoing validation.

The repository is being reorganized so that the experiment can be reproduced and evaluated more transparently. The next development priorities are:

- clearer separation of preprocessing, modeling, and evaluation code
- reproducible train/validation/test splits
- automated performance reporting
- out-of-sample and market-regime analysis
- transaction-cost and portfolio-risk evaluation

## Repository Structure

The target structure is:

```text
sage-price-model/
├── README.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── weighting.py
│   └── evaluation.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── .gitignore
```

The existing implementation should be moved into this structure only after each file's role has been verified; the reorganization itself should not change model logic.

## Tech

- Python
- NumPy / Pandas
- Matplotlib
- Statistical and machine-learning workflow

## Limitations

A strong historical fit does not establish predictive ability. Financial data are non-stationary, and model selection can introduce overfitting even when a conventional train/test split is used. The project therefore treats out-of-sample robustness, regime sensitivity, turnover, and transaction costs as first-class evaluation criteria.

## Disclaimer

This repository is an educational and research project. It is not investment advice and does not represent a guarantee of future performance.
