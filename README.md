# US Equity Options Call/Put Prediction + Historical Backtesting

## Overview
This repository contains a research-grade machine-learning system designed to scan the US equity options market and identify option contracts that have statistically favorable future return characteristics.

The prediction target is to identify whether a specific CALL or PUT option contract has attractive expected future return after accounting for moneyness, DTE, IV, Greeks, liquidity, bid/ask spread, and transaction costs.

## Initial Scope
- **Universe:** US Market (initially SPY, then QQQ, IWM)
- **Data Source:** `SaidBahaDev/options-data` (GitHub) containing historical US equity option chains (2008-2025).
- **Core Technology:** Parquet, DuckDB/Polars for out-of-core processing, Scikit-Learn/XGBoost/LightGBM for modeling.

## Project Structure
- `data/`: Raw, interim, and processed data (ignored in git).
- `configs/`: YAML configuration files.
- `src/`: Source code for data loading, feature engineering, modeling, and backtesting.
- `notebooks/`: Jupyter notebooks for EDA and prototyping.
- `tests/`: Unit tests (especially for data leakage and lookahead bias).
- `results/`: Model artifacts, backtest logs, and reports.

## Getting Started
1. Set up a virtual environment: `python -m venv venv`
2. Activate environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
3. Install dependencies: `pip install -r requirements.txt`
4. Use `src/data/download.py` to retrieve the dataset.

