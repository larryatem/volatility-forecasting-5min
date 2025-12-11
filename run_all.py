# run_all.py
"""
This is our clean, polished pipeline to run the 5-minute files contained in Project.zip.

Requirements (in venv):
  pandas, numpy, statsmodels, scikit-learn, joblib, matplotlib

What it does:
 - finds 5-minute trade & quote CSVs inside Project.zip
 - ensures 5-min log-returns exist (computes if needed)
 - computes daily RV and DRV (target)
 - aggregates microstructure features to daily level
 - creates lagged & rolling exog features
 - selects top-K features by absolute correlation
 - fits ARIMAX (SARIMAX with exog), SARIMAX, and UnobservedComponents (state-space) with exog
 - saves models, diagnostics, predictions.csv and eval_summary.csv into project_output/
"""

from pathlib import Path
import zipfile
import json
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

# functions provided in your ts_utils.py
from ts_utils import (
    compute_5min_returns,
    compute_daily_RV,
    aggregate_daily_microstructure,
    create_exog_features,
    evaluate_forecast,
    ljung_box_test,
)

warnings.filterwarnings("ignore")

# -------------------------
# Settings
# -------------------------
OUT = Path("project_output")
OUT.mkdir(exist_ok=True)

ZIP_PATH = Path("Project.zip")
TEST_DAYS = 60
TOP_K = 8
GRID_P = range(0, 3)
GRID_Q = range(0, 3)

# -------------------------
# Helpers
# -------------------------
def find_5min_files_in_zip(zip_path):
    """Return (trade_file, quote_file) filenames inside the zip that look like 5-min files."""
    with zipfile.ZipFile(zip_path, "r") as z:
        listing = [f for f in z.namelist() if not f.lower().startswith("__macosx")]
    # Candidate heuristics (ordered)
    trade_candidates = []
    quote_candidates = []
    for f in listing:
        name = f.lower()
        if not f.lower().endswith(".csv"):
            continue
        has_5min = "5min" in name or "5_min" in name or "5-min" in name or ("5" in name and "min" in name)
        if not has_5min:
            continue
        # trade-like
        if any(k in name for k in ("trade", "trades", "price", "tick")):
            trade_candidates.append(f)
        # quote-like
        if any(k in name for k in ("quote", "quotes", "microstructure", "bid", "ask")):
            quote_candidates.append(f)
    # fallback: if none matched with the above, try any csv with '5' and 'min' anywhere
    if not trade_candidates:
        for f in listing:
            name = f.lower()
            if f.lower().endswith(".csv") and "5" in name and "min" in name and any(k in name for k in ("trade","price","trades")):
                trade_candidates.append(f)
    if not quote_candidates:
        for f in listing:
            name = f.lower()
            if f.lower().endswith(".csv") and "5" in name and "min" in name and any(k in name for k in ("quote","microstructure","bid","ask")):
                quote_candidates.append(f)
    # pick the first of each candidate list (if present)
    trade_file = trade_candidates[0] if trade_candidates else None
    quote_file = quote_candidates[0] if quote_candidates else None
    return trade_file, quote_file

def read_csv_from_zip(zip_path, inner_path, **kwargs):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_path) as fh:
            return pd.read_csv(fh, **kwargs)

# -------------------------
# Main
# -------------------------
def main():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found. Place Project.zip in the same folder as run_all.py", file=sys.stderr)
        sys.exit(1)

    trade_file, quote_file = find_5min_files_in_zip(ZIP_PATH)
    if trade_file is None or quote_file is None:
        print("ERROR: Could not find 5-min trade/quote CSV files inside Project.zip (look for '5min').", file=sys.stderr)
        print("Zip contents (first 50):")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            print("\n".join(z.namelist()[:50]))
        sys.exit(1)

    print("Using trade file:", trade_file)
    print("Using quote file:", quote_file)

    trades = read_csv_from_zip(ZIP_PATH, trade_file)
    quotes = read_csv_from_zip(ZIP_PATH, quote_file)

    # ensure timestamp column exists and returns present
    # common price candidates (passed to compute_5min_returns)
    price_candidates = ['price','last','close','mid','last_price','PRICE','last_px','px_last']
    trades = compute_5min_returns(trades, timestamp_col='timestamp', price_cols=price_candidates, utc=True)

    # 1) Daily RV and DRV
    rv_df = compute_daily_RV(trades, timestamp_col='timestamp', r_col='r', utc=True)
    rv_df['DRV'] = rv_df['RV'].diff()
    rv_df = rv_df.dropna()
    rv_df.to_csv(OUT / "daily_RV.csv")

    # 2) Aggregate microstructure daily
    daily_micro = aggregate_daily_microstructure(quotes, timestamp_col='timestamp', utc=True)
    daily_micro.to_csv(OUT / "daily_micro_agg.csv")

    # 3) Merge & feature engineering
    daily = rv_df.join(daily_micro, how='inner').dropna().sort_index()
    daily.to_csv(OUT / "daily_merged.csv")

    X_base = daily.drop(columns=['RV', 'DRV'])
    X = create_exog_features(X_base, lags=[1,2,3], rolling_windows=[3,5,10])
    # de-fragment
    X = X.copy()
    combined = X.join(daily['DRV']).dropna()
    combined.to_csv(OUT / "combined_features.csv")

    # 4) Train/test split
    if len(combined) <= TEST_DAYS + 5:
        raise RuntimeError("Not enough daily observations for chosen TEST_DAYS. Reduce TEST_DAYS or provide more data.")
    train = combined.iloc[:-TEST_DAYS]
    test = combined.iloc[-TEST_DAYS:]
    y_train = train['DRV']; X_train = train.drop(columns=['DRV'])
    y_test  = test['DRV'];  X_test  = test.drop(columns=['DRV'])

    # 5) Feature selection
    corrs = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    top_feats = corrs.head(TOP_K).index.tolist()
    X_train_sel = X_train[top_feats].copy()
    X_test_sel  = X_test[top_feats].copy()
    corrs.to_csv(OUT / "feature_correlations.csv")
    with open(OUT / "chosen_exog.json", "w") as fh:
        json.dump({"top_feats": top_feats, "TOP_K": TOP_K}, fh, indent=2)

    # 6) ARIMAX (SARIMAX with exog) grid search
    print("Fitting ARIMAX (SARIMAX with exog) ...")
    best_aic_arimax = np.inf; best_res_arimax = None; best_order_arimax = None
    for p in GRID_P:
        for q in GRID_Q:
            try:
                mod = SARIMAX(endog=y_train, exog=X_train_sel, order=(p,0,q),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=200)
                if res.aic < best_aic_arimax:
                    best_aic_arimax = res.aic; best_order_arimax = (p,0,q); best_res_arimax = res
            except Exception:
                continue
    if best_res_arimax is None:
        raise RuntimeError("ARIMAX failed for all tried orders.")
    arimax_pred = best_res_arimax.get_forecast(steps=len(y_test), exog=X_test_sel).predicted_mean
    eval_arimax = evaluate_forecast(y_test.values, arimax_pred.values)
    best_res_arimax.save(OUT / "best_arimax.pickle")

    # diagnostics ARIMAX
    resid_arimax = best_res_arimax.resid
    plt.figure(figsize=(10,4)); plt.plot(resid_arimax); plt.title("ARIMAX Residuals"); plt.tight_layout(); plt.savefig(OUT/"arimax_resid.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(resid_arimax.dropna(), bins=30); plt.title("ARIMAX Residual Histogram"); plt.tight_layout(); plt.savefig(OUT/"arimax_resid_hist.png"); plt.close()
    ljung_box_test(resid_arimax).to_csv(OUT / "ljungbox_arimax.csv")

    # 7) SARIMAX
    print("Fitting SARIMAX ...")
    best_aic = np.inf; best_res = None; best_order = None
    for p in GRID_P:
        for q in GRID_Q:
            try:
                mod = SARIMAX(endog=y_train, exog=X_train_sel, order=(p,0,q),
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False, maxiter=200)
                if res.aic < best_aic:
                    best_aic = res.aic; best_order = (p,0,q); best_res = res
            except Exception:
                continue
    if best_res is None:
        raise RuntimeError("SARIMAX failed for all tried orders.")
    sarimax_pred = best_res.get_forecast(steps=len(y_test), exog=X_test_sel).predicted_mean
    eval_sarimax = evaluate_forecast(y_test.values, sarimax_pred.values)
    best_res.save(OUT / "best_sarimax.pickle")

    resid_sarimax = best_res.resid
    plt.figure(figsize=(10,4)); plt.plot(resid_sarimax); plt.title("SARIMAX Residuals"); plt.tight_layout(); plt.savefig(OUT/"sarimax_resid.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(resid_sarimax.dropna(), bins=30); plt.title("SARIMAX Residual Histogram"); plt.tight_layout(); plt.savefig(OUT/"sarimax_resid_hist.png"); plt.close()
    ljung_box_test(resid_sarimax).to_csv(OUT / "ljungbox_sarimax.csv")

    # 8) State-space (UnobservedComponents) WITH exog
    print("Fitting UnobservedComponents (state-space) with exog ...")
    uc = UnobservedComponents(endog=y_train, level='local level', exog=X_train_sel)
    res_uc = uc.fit(disp=False, maxiter=200)
    uc_pred = res_uc.get_forecast(steps=len(y_test), exog=X_test_sel).predicted_mean
    eval_uc = evaluate_forecast(y_test.values, uc_pred.values)
    res_uc.save(OUT / "res_uc.pickle")

    resid_uc_train = (y_train - res_uc.fittedvalues).dropna()
    plt.figure(figsize=(10,4)); plt.plot(resid_uc_train); plt.title("UC Train Residuals"); plt.tight_layout(); plt.savefig(OUT/"uc_resid_train.png"); plt.close()
    plt.figure(figsize=(6,4)); plt.hist(resid_uc_train.dropna(), bins=30); plt.title("UC Residual Histogram (train)"); plt.tight_layout(); plt.savefig(OUT/"uc_resid_hist.png"); plt.close()
    ljung_box_test(resid_uc_train).to_csv(OUT / "ljungbox_uc_train.csv")

    # 9) Save predictions + evaluation
    preds_df = pd.DataFrame({
        'y_test': y_test.values,
        'arimax_pred': arimax_pred.values,
        'sarimax_pred': sarimax_pred.values,
        'uc_pred': uc_pred.values
    }, index=y_test.index)
    preds_df.to_csv(OUT / "predictions.csv")

    eval_df = pd.DataFrame([eval_arimax, eval_sarimax, eval_uc], index=['ARIMAX','SARIMAX','UC'])
    eval_df.to_csv(OUT / "eval_summary.csv")

    # metadata
    joblib.dump({
        'top_feats': top_feats,
        'best_order_sarimax': best_order,
        'best_order_arimax': best_order_arimax,
        'best_aic_sarimax': best_aic,
        'best_aic_arimax': best_aic_arimax,
        'TEST_DAYS': TEST_DAYS
    }, OUT / "model_meta.joblib")

    with open(OUT / "readme_assumptions.json", "w") as fh:
        json.dump({
            'RV_formula': 'sum of squared 5-min log returns per day',
            'DRV_formula': 'RV_t - RV_{t-1}',
            'micro_aggregation': 'daily mean/std for spreads/mid, sum/mean for OFI and quote sizes',
            'test_days': TEST_DAYS,
            'top_k_features': TOP_K,
            'best_order_sarimax': str(best_order),
            'best_order_arimax': str(best_order_arimax)
        }, fh, indent=2)

    # final prints
    print("All done. Outputs saved to", OUT.resolve())
    print("Top features:", top_feats)
    print("ARIMAX eval:", eval_arimax)
    print("SARIMAX eval:", eval_sarimax)
    print("UC eval:", eval_uc)


if __name__ == "__main__":
    main()
