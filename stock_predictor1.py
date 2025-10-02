# Machine Learning Model for Stock Prices Predictor (Streamlit-Compatible Final Version)
# Predicts if closing price will increase the next day using S&P 500 data.
# Implements technical indicators manually. Fixed: Timezone-naive index, broadcasting issues, Tkinter warnings, backtesting.

import os
import logging
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent Tkinter GUI issues
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

# -------------------- Config & Setup -------------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

TICKER = "^GSPC"
START_DATE = '1990-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_FILE = 'sp500.csv'
MODEL_FILE = 'stock_predictor_model.pkl'
PLOT_DIR = 'plots'
THRESHOLD = 0.6  # Probability threshold

os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------- Helper Functions -------------------- #

def make_naive_datetime_index(df):
    """Ensure the index is timezone-naive datetime."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

def fetch_stock_data(ticker=TICKER, filename=DATA_FILE, start=START_DATE, end=END_DATE):
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df = make_naive_datetime_index(df)
        logging.info(f"Loaded data from {filename} ({len(df)} rows)")
    else:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = make_naive_datetime_index(df)
            df.to_csv(filename)
            logging.info(f"Downloaded and saved {len(df)} rows for {ticker}")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            dates = pd.date_range(start=pd.to_datetime(start), end=pd.to_datetime(end), freq='B')
            np.random.seed(42)
            closes = 1000 + np.cumsum(np.random.randn(len(dates)) * 2)
            df = pd.DataFrame({
                'Close': closes,
                'Open': closes*0.99 + np.random.randn(len(dates))*0.1,
                'High': closes*1.01 + np.abs(np.random.randn(len(dates))*0.5),
                'Low': closes*0.99 - np.abs(np.random.randn(len(dates))*0.5),
                'Volume': np.random.randint(1e6, 1e9, len(dates))
            }, index=dates)
            logging.warning("Using dummy data")
    logging.info(f"Data range: {df.index.min()} to {df.index.max()}")
    return df

def prepare_data(df):
    df = make_naive_datetime_index(df)
    for col in ['Dividends', 'Stock Splits']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df.dropna(subset=['Target'], inplace=True)
    logging.info(f"Data after cleaning: {len(df)} rows, Target distribution: {df['Target'].value_counts().to_dict()}")
    return df

# -------------------- Technical Indicators -------------------- #

def manual_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def manual_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.fillna(0), signal_line.fillna(0)

def manual_bollinger_bands(close, window=20, num_std=2):
    ma = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std().fillna(0)
    upper = ma + num_std*std
    lower = ma - num_std*std
    return upper, lower

def add_technical_indicators(df):
    """Add all technical indicators safely for Streamlit."""
    df = make_naive_datetime_index(df)
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['High_Low_Ratio'] = (df['High']/df['Low']).fillna(1)
    df['Open_Close_Ratio'] = (df['Close']/df['Open']).fillna(1)

    horizons = [2,5,20,60,250]
    for h in horizons:
        if len(df) >= h:
            ma = df['Close'].rolling(h, min_periods=1).mean()
            df[f'Close_MA_{h}'] = ma.fillna(df['Close'])
            df[f'Close_Ratio_{h}'] = (df['Close'] / df[f'Close_MA_{h}']).fillna(1.0)
            # Use rolling sum of past Target safely
            if 'Target' in df.columns:
                trend = df['Target'].shift(1).rolling(h, min_periods=1).sum()
                df[f'Trend_{h}'] = trend.fillna(trend.mean())
    df['RSI'] = manual_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = manual_macd(df['Close'])
    df['BB_High'], df['BB_Low'] = manual_bollinger_bands(df['Close'])
    bb_width = df['BB_High'] - df['BB_Low']
    df['BB_Position'] = np.where(bb_width > 0, (df['Close'] - df['BB_Low']) / bb_width, 0.5)
    df['Volume_MA_20'] = df['Volume'].rolling(20, min_periods=1).mean().fillna(df['Volume'])
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    df = df.dropna(thresh=int(len(df.columns)*0.8))
    logging.info(f"Indicators added: {len(df)} rows")
    return df

def get_predictors(df):
    base = ['Close','Volume','Open','High','Low','Returns','High_Low_Ratio','Open_Close_Ratio']
    indicators = [c for c in df.columns if any(k in c for k in ['Close_Ratio_','Trend_','RSI','MACD','BB_','Volume_Ratio'])]
    predictors = [p for p in base+indicators if p in df.columns and (df[p].notna().sum()/len(df))>0.9]
    logging.info(f"Using {len(predictors)} predictors (example: {predictors[:5]})")
    return predictors

# -------------------- Modeling -------------------- #

def train_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1,n_jobs=-1))
    ])
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__min_samples_split': [50, 100],
        'classifier__max_depth': [10, 20, None]
    }
    n_splits = min(5, len(X)//500)
    tscv = TimeSeriesSplit(n_splits=max(2,n_splits))
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='precision', n_jobs=-1, verbose=1)
    grid.fit(X, y)
    logging.info(f"Best params: {grid.best_params_}, CV Precision: {grid.best_score_:.4f}")
    joblib.dump({'model': grid.best_estimator_, 'predictors': X.columns.tolist()}, MODEL_FILE)
    return grid.best_estimator_

# -------------------- Backtest & Latest Predictions -------------------- #

def backtest(data, model, predictors, start=2500, step=250, threshold=THRESHOLD):
    all_preds=[]
    max_idx = len(data)
    for i in range(start, max_idx, step):
        end_i = min(i+step, max_idx)
        train = data.iloc[:i].copy()
        test = data.iloc[i:end_i].copy()
        train_clean = train.dropna(subset=predictors+['Target'])
        test_clean = test.dropna(subset=predictors)
        if len(train_clean)<50 or len(test_clean)<5: continue
        model.fit(train_clean[predictors],train_clean['Target'])
        probs = model.predict_proba(test_clean[predictors])[:,1]
        preds = (probs>=threshold).astype(int)
        df_preds = pd.DataFrame({'Target':test.loc[test_clean.index,'Target'].values,
                                 'Predictions':preds,'Probs':probs},index=test_clean.index)
        all_preds.append(df_preds)
    if all_preds:
        return pd.concat(all_preds)
    else:
        logging.warning("No backtest windows")
        return pd.DataFrame(columns=['Target','Predictions','Probs'])

def predict_latest(model, data, predictors, n_days=5, threshold=THRESHOLD):
    if len(data) < n_days + 100: return None
    train_size = len(data)-n_days
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    train_clean = train.dropna(subset=predictors+['Target'])
    test_clean = test.dropna(subset=predictors)
    if len(train_clean)<50 or len(test_clean)==0: return None
    model.fit(train_clean[predictors],train_clean['Target'])
    probs = model.predict_proba(test_clean[predictors])[:,1]
    preds = (probs>=threshold).astype(int)
    latest_df = pd.DataFrame({
        'Date': test_clean.index,
        'Close': test_clean['Close'].values,
        'Target': test.loc[test_clean.index,'Target'].values,
        'Predictions': preds,'Probs':probs})
    return latest_df

# -------------------- Plotting -------------------- #

def plot_save_close(df,filename='closing_price.png',ticker=TICKER):
    plt.figure(figsize=(14,7))
    df['Close'].plot(title=f'{ticker} Closing Price')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.savefig(os.path.join(PLOT_DIR,filename),dpi=300,bbox_inches='tight')
    plt.close()

def plot_predictions(preds, title="Predictions"):
    if len(preds)>0:
        plt.figure(figsize=(14,7))
        preds[['Target','Predictions']].plot(title=title)
        plt.ylabel('Target (1=Up,0=Down)')
        plt.xlabel('Date')
        plt.savefig(os.path.join(PLOT_DIR,f'predictions_{title.lower().replace(" ","_")}.png'),dpi=300,bbox_inches='tight')
        plt.close()

def plot_feature_importance(model,predictors):
    try: importances=model.named_steps['classifier'].feature_importances_
    except: importances=model.feature_importances_
    feat_imp=pd.Series(importances,index=predictors).sort_values(ascending=False)
    plt.figure(figsize=(10,8))
    feat_imp.head(10).plot(kind='barh',title='Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,'feature_importance.png'),dpi=300,bbox_inches='tight')
    plt.close()

# -------------------- Evaluation -------------------- #

def evaluate_model(y_true,y_pred,y_prob,title="Evaluation"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true,y_pred))
    metrics={'Precision':precision_score(y_true,y_pred,zero_division=0),
             'Recall':recall_score(y_true,y_pred,zero_division=0),
             'F1-Score':f1_score(y_true,y_pred,zero_division=0),
             'Accuracy':accuracy_score(y_true,y_pred),
             'ROC-AUC':roc_auc_score(y_true,y_prob)}
    for k,v in metrics.items(): print(f"{k}: {v:.4f}")
    cm=confusion_matrix(y_true,y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Down','Up'],yticklabels=['Down','Up'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.savefig(os.path.join(PLOT_DIR,f'confusion_matrix_{title.lower().replace(" ","_")}.png'),dpi=300,bbox_inches='tight')
    plt.close()
    return metrics
