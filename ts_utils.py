# ts_utils.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox

def _find_price_col(df):
    """Return a sensible price column name from common names or None."""
    candidates = ['r','return','ret','log_return','price','last','close','mid']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_5min_returns(trades_df, timestamp_col='timestamp', price_cols=None, utc=True):
    """
    Ensure there is a column 'r' = log-return at 5-min resolution.
    If the input is already 5-min and contains a return column, we'll use it.
    Otherwise we try to compute r = log(price) - log(price.shift(1)).
    Returns a copy with 'timestamp' datetime and column 'r'.
    """
    t = trades_df.copy()
    if utc:
        t[timestamp_col] = pd.to_datetime(t[timestamp_col], utc=True)
    else:
        t[timestamp_col] = pd.to_datetime(t[timestamp_col])
    # Try common return column names
    possible_r = [c for c in ['r','return','ret','log_return'] if c in t.columns]
    if possible_r:
        t['r'] = pd.to_numeric(t[possible_r[0]], errors='coerce')
        return t
    # Otherwise compute from price column
    price_col = None
    if price_cols is not None:
        for c in price_cols:
            if c in t.columns:
                price_col = c
                break
    if price_col is None:
        price_col = _find_price_col(t)
    if price_col is None:
        raise RuntimeError("Could not find price or return column in trades data. Provide price column.")
    # Convert price and compute log returns (assumes rows are 5-min aggregated ticks)
    t[price_col] = pd.to_numeric(t[price_col], errors='coerce')
    t = t.sort_values(timestamp_col)
    t['logp'] = np.log(t[price_col])
    t['r'] = t['logp'].diff()
    # first row will be NaN — that's OK
    return t

def compute_daily_RV(trades_df, timestamp_col='timestamp', r_col='r', utc=True):
    """
    Compute daily Realized Volatility (RV) as sum of squared 5-min log returns per day.
    Returns DataFrame indexed by date with column RV.
    """
    t = trades_df.copy()
    if utc:
        t[timestamp_col] = pd.to_datetime(t[timestamp_col], utc=True)
    else:
        t[timestamp_col] = pd.to_datetime(t[timestamp_col])
    # ensure r is numeric
    t[r_col] = pd.to_numeric(t[r_col], errors='coerce')
    # drop missing returns
    t = t.dropna(subset=[r_col])
    t['date'] = t[timestamp_col].dt.date
    t['r_sq'] = t[r_col] ** 2
    rv = t.groupby('date', as_index=True)['r_sq'].sum().rename('RV')
    rv.index = pd.to_datetime(rv.index)
    return rv.to_frame()

def aggregate_daily_microstructure(quotes_df, timestamp_col='timestamp', agg_map=None, utc=True):
    """
    Aggregate microstructure features from quote-level 5-min data to daily level.
    Default behavior:
      - OFI, Q_bid, Q_ask aggregated as sum and mean
      - others aggregated as mean and std
    """
    q = quotes_df.copy()
    if utc:
        q[timestamp_col] = pd.to_datetime(q[timestamp_col], utc=True)
    else:
        q[timestamp_col] = pd.to_datetime(q[timestamp_col])
    q['date'] = q[timestamp_col].dt.date
    default_feats = ['s','m','OFI','dmt','s_rel','Q_bid','Q_ask','B','A','bid','ask','size']
    cols = [c for c in default_feats if c in q.columns]
    if agg_map is None:
        agg_map = {}
        for c in cols:
            if c in ['OFI','Q_bid','Q_ask','size']:
                agg_map[c] = ['sum','mean']
            else:
                agg_map[c] = ['mean','std']
    daily = q.groupby('date').agg(agg_map)
    # flatten columns
    daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
    daily.index = pd.to_datetime(daily.index)
    return daily

def create_exog_features(daily_micro_df, lags=[1,2,3], rolling_windows=[3,5,10]):
    """
    Given a daily microstructure dataframe, create lag and rolling-window features.
    Returns a new DataFrame of features aligned to date index.
    """
    X = daily_micro_df.copy()
    for col in list(daily_micro_df.columns):
        for l in lags:
            X[f"{col}_lag{l}"] = X[col].shift(l)
        for w in rolling_windows:
            X[f"{col}_rm{w}"] = X[col].rolling(w).mean()
            X[f"{col}_rstd{w}"] = X[col].rolling(w).std()
    return X

def evaluate_forecast(y_true, y_pred):
    """
    Returns dict with rmse, mae, mape
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # use small epsilon to avoid div by zero
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-12))) * 100
    return dict(rmse=float(rmse), mae=float(mae), mape=float(mape))

def ljung_box_test(residuals, lags=[10,20]):
    """
    Return DataFrame of Ljung-Box test results for given lags.
    """
    res = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    return res
