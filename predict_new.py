# predict_new.py
# Load trained Stock Predictor and make predictions on recent/new data

import os
import logging
import warnings
from datetime import datetime
import pandas as pd
import joblib
import yfinance as yf

# -------------------- Config -------------------- #
MODEL_FILE = 'stock_predictor_model.pkl'
PLOT_DIR = 'plots'
TICKER = "^GSPC"
N_DAYS = 5
THRESHOLD = 0.6

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------- Helper -------------------- #
def make_naive_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

def fetch_recent_data(ticker=TICKER, days=30):
    """Fetch last `days` of stock data."""
    try:
        stock = yf.Ticker(ticker)
        end = datetime.now()
        start = end - pd.Timedelta(days=days)
        df = stock.history(start=start, end=end)
        if df.empty:
            raise ValueError("No data fetched")
        df = make_naive_datetime_index(df)
        logging.info(f"Fetched {len(df)} rows of recent data")
        return df
    except Exception as e:
        logging.error(f"Error fetching recent data: {e}")
        return None

# -------------------- Prediction -------------------- #
def predict_new_data(model_file=MODEL_FILE, n_days=N_DAYS, threshold=THRESHOLD):
    if not os.path.exists(model_file):
        logging.error(f"Model file not found: {model_file}")
        return
    
    # Load model (pipeline with scaler + classifier)
    model = joblib.load(model_file)
    
    # Extract predictors from model
    if hasattr(model, 'named_steps'):
        if 'classifier' in model.named_steps:
            try:
                predictors = model.feature_names_in_
            except:
                predictors = None
        else:
            predictors = None
    else:
        predictors = None

    if predictors is None:
        logging.error("Cannot retrieve predictors from model. Re-run stock_predictor.py to save with predictors.")
        return
    
    # Fetch recent data
    df = fetch_recent_data(days=n_days + 10)
    if df is None or len(df) < n_days:
        logging.error("Not enough recent data to predict")
        return
    
    # Prepare features (must match those used in training)
    df = make_naive_datetime_index(df)
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['High_Low_Ratio'] = (df['High'] / df['Low']).fillna(1)
    df['Open_Close_Ratio'] = (df['Close'] / df['Open']).fillna(1)
    
    # For any missing predictor columns, fill with 0
    for col in predictors:
        if col not in df.columns:
            df[col] = 0
    
    X = df[predictors].iloc[-n_days:]
    
    # Predict
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    result = pd.DataFrame({
        'Date': X.index,
        'Close': df['Close'].iloc[-n_days:].values,
        'Predictions': preds,
        'Probability_Up': probs
    })
    
    print(f"\nPredictions for last {n_days} days:\n")
    print(result.round(4).to_string(index=False))

    return result

# -------------------- Main -------------------- #
if __name__ == "__main__":
    predict_new_data()
