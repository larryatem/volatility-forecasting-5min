# Intraday Volatility Forecasting Using Microstructure Features

### Author: Larry Kevin Atem Nkatcha
Last Updated: December 2025

# Overview

This project builds a complete intraday volatility forecasting pipeline using 5-minute trade and quote data.
The objective is to predict Daily Realized Volatility (RV) and Daily Realized Volatility Changes (DRV) using both:

ARIMAX (ARIMA with exogenous microstructure features)

SARIMAX (Seasonal ARIMAX)

State-Space / Unobserved Components Model with microstructure regressors

This repository demonstrates the entire workflow a quant researcher would build:
data cleaning, feature engineering, volatility computation, model selection, forecasting, and diagnostics.

# Project Structure
project/
│
├── run_all.py
├── ts_utils.py
│
├── Project.zip
│
├── project_output/
│   ├── daily_RV.csv
│   ├── daily_micro_agg.csv
│   ├── daily_merged.csv
│   ├── combined_features.csv
│   ├── chosen_exog.json
│   ├── predictions.csv
│   ├── eval_summary.csv
│   ├── eval_summary_tabular.csv
│   ├── readme_assumptions.json
│   ├── model_meta.joblib
│   ├── best_arimax.pickle
│   ├── best_sarimax.pickle
│   ├── res_uc.pickle
│   ├── residual plots (.png)
│   ├── ljungbox diagnostics (.csv)
│   └── forecast_comparison.png
│
└── README.md

# Data Inputs

The raw data (inside Project.zip) includes:

5-minute trade data (prices → returns → realized volatility)

5-minute quote data (book depths, OFI, spreads, mid-price, etc.)

Both datasets are merged and transformed into daily aggregates and microstructure features.

# Methodology
## 1. Realized Volatility

Daily RV is computed from squared intraday returns:

RV_t = Σ rₜ,ᵢ²


Daily RV difference (DRV):

DRV_t = RV_t – RV_{t-1}

## 2. Microstructure Features

From 5-minute quote data, the pipeline constructs features including:

Bid/ask spread (mean, std)

Mid-price volatility

Order Flow Imbalance (OFI)

Depth variables: Q_bid, Q_ask, B, A

Rolling windows (3, 5, 10 days)

Lagged features (1–3 day lags)

These form the exogenous regressor matrix used by all models.

## 3. Modeling

All models forecast DRV using identical exogenous features.

ARIMAX: small grid search over (p,0,q).
SARIMAX: seasonal ARIMA with selected exogenous regressors.
Unobserved Components Model: state-space model with local-level trend + microstructure terms.

## 4. Evaluation

Out-of-sample predictions are compared using:

RMSE

MAE

MAPE

Residual autocorrelation (Ljung–Box)

Residual distribution

Forecast comparison plots

Outputs stored in:
eval_summary.csv and predictions.csv

# Key Output Files

These are the most important files for reviewing results:

forecast_comparison.png – actual vs forecasted DRV

feature_correlations.csv – exogenous feature importance

arimax_resid.png, sarimax_resid.png, uc_resid_hist.png – residual diagnostics

ljungbox_*.csv – autocorrelation tests

eval_summary.csv – cross-model comparison

predictions.csv – forecast values

# Running the Pipeline

Run the full workflow with:

python3 run_all.py


The script will automatically:

Load 5-minute trade/quote data

Compute realized volatility

Build microstructure exogenous features

Fit ARIMAX, SARIMAX, and State-Space models

Generate forecasts

Save diagnostics and summary statistics

All outputs appear in project_output/.

# Python Requirements
numpy<2
pandas
statsmodels
scikit-learn
joblib
matplotlib

# Summary of Findings

Microstructure features significantly improve DRV forecasting accuracy.

OFI, spreads, and depth-related volatility are consistently the strongest predictors.

ARIMAX and SARIMAX perform similarly; UC model provides smoother trend extraction.

Diagnostics confirm reduced residual autocorrelation when microstructure features are included.

# Why This Project Matters

This repository demonstrates:

Handling real intraday trade/quote microstructure data

Realized volatility construction

Statistical time-series modeling (ARIMAX/SARIMAX/State-Space)

Pipeline automation in Python

Model selection, diagnostics, and forecasting

Skills relevant for quant research, quant analyst, trading, and risk roles
