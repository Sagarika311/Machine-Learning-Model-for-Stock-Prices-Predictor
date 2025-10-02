# stock_predictor.py
# Machine Learning Model for Stock Prices Predictor (Final Fixed Version)
# Predicts if closing price will increase the next day using S&P 500 data.
# Implements technical indicators manually. Fixed: Timezone conversion, Index.tz errors.

import os
import logging
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import yfinance as yf

# Setup logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configuration
TICKER = "^GSPC"  # S&P 500
START_DATE = '1990-01-01'  # String, converted to naive Timestamp
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_FILE = 'sp500.csv'
MODEL_FILE = 'stock_predictor_model.pkl'
PLOT_DIR = 'plots'
THRESHOLD = 0.6  # Probability threshold for predictions

# Create plot directory
os.makedirs(PLOT_DIR, exist_ok=True)

def make_naive_datetime_index(df):
    """Helper: Convert index to timezone-naive DatetimeIndex safely."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

def fetch_stock_data(ticker=TICKER, filename=DATA_FILE, start=START_DATE, end=END_DATE):
    """Fetch and cache stock data, ensuring timezone-naive DatetimeIndex."""
    if os.path.exists(filename):
        # Load CSV with parse_dates for index
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df = make_naive_datetime_index(df)
        logging.info(f"Loaded data from {filename} ({len(df)} rows, naive index: {not hasattr(df.index, 'tz') or df.index.tz is None})")
    else:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end)
            if df.empty or len(df) < 100:
                raise ValueError(f"Insufficient data fetched for {ticker} ({len(df)} rows).")
            # Make naive
            df = make_naive_datetime_index(df)
            df.to_csv(filename)
            logging.info(f"Downloaded and saved {len(df)} rows for {ticker} to {filename}")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            # Fallback dummy data (naive index)
            dates = pd.date_range(start=pd.to_datetime(start, utc=True).tz_localize(None), 
                                  end=pd.to_datetime(end, utc=True).tz_localize(None), freq='B')
            np.random.seed(42)
            closes = 1000 + np.cumsum(np.random.randn(len(dates)) * 2)
            df = pd.DataFrame({
                'Close': closes,
                'Open': closes * 0.99 + np.random.randn(len(dates)) * 0.1,
                'High': closes * 1.01 + np.abs(np.random.randn(len(dates)) * 0.5),
                'Low': closes * 0.99 - np.abs(np.random.randn(len(dates)) * 0.5),
                'Volume': np.random.randint(1e6, 1e9, len(dates))
            }, index=dates)
            logging.warning("Using dummy data. Connect to internet for real S&P 500 data.")
    
    # Final log
    logging.info(f"Data range: {df.index.min()} to {df.index.max()} (naive: {not hasattr(df.index, 'tz') or df.index.tz is None}, type: {type(df.index)})")
    return df

def prepare_data(df):
    """Clean and prepare data: create target and basic features."""
    # Ensure naive index
    df = make_naive_datetime_index(df)
    
    # Clean columns
    for col in ['Dividends', 'Stock Splits']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # Create target
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    
    # Slice from START_DATE (naive Timestamp)
    start_ts = pd.to_datetime(START_DATE, utc=True).tz_localize(None)
    df = df[df.index >= start_ts].copy()
    
    if len(df) == 0:
        logging.warning(f"No data >= {start_ts}. Using all data.")
    
    # Drop NaNs in target
    initial_len = len(df)
    df.dropna(subset=['Target'], inplace=True)
    logging.info(f"Data after cleaning: {len(df)} rows (dropped {initial_len - len(df)})")
    
    if len(df) < 1000:
        raise ValueError(f"Insufficient data: {len(df)} rows.")
    
    logging.info(f"Target balance: {df['Target'].value_counts().to_dict()}")
    return df

def manual_rsi(close, window=14):
    """Manual RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid div by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def manual_macd(close, fast=12, slow=26, signal=9):
    """Manual MACD."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.fillna(0), signal_line.fillna(0)

def manual_bollinger_bands(close, window=20, num_std=2):
    """Manual Bollinger Bands."""
    rolling_mean = close.rolling(window=window, min_periods=1).mean()
    rolling_std = close.rolling(window=window, min_periods=1).std().fillna(0)
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def add_technical_indicators(df):
    """Add indicators with robust NaN handling."""
    df = make_naive_datetime_index(df)  # Ensure index
    
    # Basic
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['High_Low_Ratio'] = (df['High'] / df['Low']).fillna(1.0)
    df['Open_Close_Ratio'] = (df['Close'] / df['Open']).fillna(1.0)
    
    # MAs
    horizons = [2, 5, 20, 60, 250]
    for h in horizons:
        if len(df) >= h:
            ma = df['Close'].rolling(h, min_periods=1).mean()
            df[f'Close_MA_{h}'] = ma.fillna(df['Close'])
            df[f'Close_Ratio_{h}'] = df['Close'] / df[f'Close_MA_{h}']
            trend = df['Target'].shift(1).rolling(h, min_periods=1).sum()
            df[f'Trend_{h}'] = trend.fillna(trend.mean())
    
    # RSI
    df['RSI'] = manual_rsi(df['Close'])
    
    # MACD
    df['MACD'], df['MACD_Signal'] = manual_macd(df['Close'])
    
    # BB
    df['BB_High'], df['BB_Low'] = manual_bollinger_bands(df['Close'])
    bb_width = df['BB_High'] - df['BB_Low']
    df['BB_Position'] = np.where(bb_width > 0, (df['Close'] - df['BB_Low']) / bb_width, 0.5)
    
    # Volume
    df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().fillna(df['Volume'])
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Drop incomplete rows
    initial_len = len(df)
    df = df.dropna(thresh=int(len(df.columns) * 0.8))
    logging.info(f"Indicators added: {len(df)} rows (dropped {initial_len - len(df)})")
    
    if len(df) < 500:
        raise ValueError(f"Too few rows after indicators: {len(df)}.")
    
    return df

def get_predictors(df):
    """Safe predictor selection."""
    base = ['Close', 'Volume', 'Open', 'High', 'Low', 'Returns', 'High_Low_Ratio', 'Open_Close_Ratio']
    indicators = [col for col in df.columns if any(ind in col for ind in ['Close_Ratio_', 'Trend_', 'RSI', 'MACD', 'BB_', 'Volume_Ratio'])]
    predictors = [p for p in base + indicators if p in df.columns and (df[p].notna().sum() / len(df)) > 0.9]
    
    if len(predictors) < 5:
        predictors = [p for p in base if p in df.columns][:5]
        logging.warning(f"Limited to {len(predictors)} basic predictors.")
    
    logging.info(f"Using {len(predictors)} predictors (e.g., {predictors[:5]})")
    return predictors

def train_model(X, y):
    """Train model with CV."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__min_samples_split': [50, 100],
        'classifier__max_depth': [10, 20, None]
    }
    
    n_splits = min(5, len(X) // 500)
    tscv = TimeSeriesSplit(n_splits=max(2, n_splits))  # At least 2 splits
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='precision', n_jobs=-1, verbose=1)
    
    grid_search.fit(X, y)
    
    logging.info(f"Best params: {grid_search.best_params_}")
    logging.info(f"Best CV Precision: {grid_search.best_score_:.4f}")
    
    # Save model
    joblib.dump(grid_search.best_estimator_, MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")
    
    return grid_search.best_estimator_

def backtest(data, model, predictors, start=2500, step=250, threshold=THRESHOLD):
    """Rolling backtest."""
    all_preds = []
    max_idx = len(data)
    
    for i in range(start, max_idx, step):
        end_i = min(i + step, max_idx)
        train = data.iloc[:i].copy()
        test = data.iloc[i:end_i].copy()
        
        if len(train) < 100 or len(test) < 10:
            continue
        
        train_clean = train.dropna(subset=predictors + ['Target'])
        test_clean = test.dropna(subset=predictors)
        
        if len(train_clean) < 50 or len(test_clean) < 5:
            continue
        
        model.fit(train_clean[predictors], train_clean['Target'])
        probs = model.predict_proba(test_clean[predictors])[:, 1]
        preds = (probs >= threshold).astype(int)
        
        pred_df = pd.DataFrame({
            'Target': test.loc[test_clean.index, 'Target'].values,
            'Predictions': preds,
            'Probs': probs
        }, index=test_clean.index)
        
        all_preds.append(pred_df)
    
    if all_preds:
        result = pd.concat(all_preds)
        logging.info(f"Backtest: {len(result)} predictions")
        return result
    else:
        logging.warning("No backtest windows; skipping.")
        return pd.DataFrame(columns=['Target', 'Predictions', 'Probs'])

def evaluate_model(y_true, y_pred, y_prob, title="Test Set"):
    """Metrics and plot."""
    print(f"\n=== {title} Evaluation ===")
    print(classification_report(y_true, y_pred))
    
    metrics = {
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    
    for name, score in metrics.items():
        print(f"{name}: {score:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{title.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics

def plot_predictions(predictions, title="Backtest"):
    """Plot predictions."""
    if len(predictions) > 0:
        plt.figure(figsize=(14, 7))
        predictions[['Target', 'Predictions']].plot(title=f'{title} - Actual vs Predicted')
        plt.ylabel('Target (1=Up, 0=Down)')
        plt.xlabel('Date')
        plt.legend(['Actual', 'Predicted'])
        plt.savefig(os.path.join(PLOT_DIR, f'predictions_{title.lower().replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        logging.warning(f"No data to plot for {title}.")

def plot_feature_importance(model, predictors):
    """Feature importance plot."""
    try:
        importances = model.named_steps['classifier'].feature_importances_
    except:
        importances = model.feature_importances_
    
    feature_imp = pd.Series(importances, index=predictors).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    feature_imp.head(10).plot(kind='barh', title='Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def predict_latest(model, data, predictors, n_days=5, threshold=THRESHOLD):
    """Latest predictions."""
    if len(data) < n_days + 100:
        return None
    
    train_size = len(data) - n_days
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    
    train_clean = train.dropna(subset=predictors + ['Target'])
    test_clean = test.dropna(subset=predictors)
    
    if len(train_clean) < 50 or len(test_clean) == 0:
        return None
    
    model.fit(train_clean[predictors], train_clean['Target'])
    probs = model.predict_proba(test_clean[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    
    latest_df = pd.DataFrame({
        'Date': test_clean.index,
        'Close': test_clean['Close'].values,
        'Target': test.loc[test_clean.index, 'Target'].values,
        'Predictions': preds,
        'Probs': probs
    })
    
    print(f"\nLatest Predictions (Last {len(latest_df)} Days):\n")
    print(latest_df.round(4).to_string(index=False))
    return latest_df

def plot_closing_price(df, ticker=TICKER):
    """Plot historical closing price."""
    plt.figure(figsize=(14, 7))
    df['Close'].plot(title=f'{ticker} Closing Price Over Time')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.savefig(os.path.join(PLOT_DIR, 'closing_price.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main pipeline."""
    try:
        logging.info("Starting Stock Predictor Pipeline...")
        
        # 1. Fetch data
        df = fetch_stock_data()
        
        # 2. Prepare data
        df = prepare_data(df)
        plot_closing_price(df)
        
        # 3. Add indicators
        df = add_technical_indicators(df)
        
        # 4. Predictors
        predictors = get_predictors(df)
        
                # 5. Split (80/20 time-based)
        split_idx = int(len(df) * 0.8)
        X_train = df[predictors].iloc[:split_idx].dropna()
        y_train = df['Target'].iloc[:split_idx].loc[X_train.index]
        X_test = df[predictors].iloc[split_idx:].dropna()
        y_test = df['Target'].iloc[split_idx:].loc[X_test.index]
        
        logging.info(f"Train/Test split: {len(X_train)} / {len(X_test)} rows")
        
        # 6. Train model
        model = train_model(X_train, y_train)
        
        # 7. Backtest on train data
        backtest_data = df.iloc[:split_idx]
        predictions = backtest(backtest_data, model, predictors)
        plot_predictions(predictions, "Train Backtest")
        
        # 8. Test set predictions
        if len(X_test) > 0:
            test_probs = model.predict_proba(X_test)[:, 1]
            test_preds = (test_probs >= THRESHOLD).astype(int)
            
            # 9. Evaluate test set
            test_metrics = evaluate_model(y_test, test_preds, test_probs, "Test Set")
        else:
            logging.warning("No test data available; skipping evaluation.")
            test_metrics = {}
        
        # 10. Feature importance
        plot_feature_importance(model, predictors)
        
        # 11. Latest predictions (last 5 days)
        latest = predict_latest(model, df, predictors, n_days=5)
        if latest is not None:
            plot_predictions(latest, "Latest 5 Days")
        
        logging.info("Pipeline completed successfully!")
        if test_metrics:
            logging.info(f"Overall Test Precision: {test_metrics['Precision']:.4f}")
        print("\n" + "="*50)
        print("STOCK PREDICTOR COMPLETE!")
        print(f"Model saved: {MODEL_FILE}")
        print(f"Plots saved in: {PLOT_DIR}/")
        print("="*50)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
