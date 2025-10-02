# stock_predictor.py
import os
import logging
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILE = 'stock_predictor_model.pkl'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)
THRESHOLD = 0.6

# ---------------- Data Utilities ---------------- #

def make_naive_datetime_index(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

def fetch_stock_data(ticker="^GSPC", filename="sp500.csv", start="1990-01-01", end=datetime.now().strftime('%Y-%m-%d')):
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        df = make_naive_datetime_index(df)
        logging.info(f"Loaded data from {filename} ({len(df)} rows)")
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
        df = make_naive_datetime_index(df)
        df.to_csv(filename)
        logging.info(f"Downloaded {len(df)} rows for {ticker}")
    return df

def prepare_data(df):
    df = make_naive_datetime_index(df)
    for col in ['Dividends', 'Stock Splits']:
        if col in df.columns: df.drop(col, axis=1, inplace=True)
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    df.dropna(subset=['Target'], inplace=True)

    # Additional Features
    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['High_Low_Ratio'] = (df['High'] / df['Low']).fillna(1)
    df['Open_Close_Ratio'] = (df['Open'] / df['Close']).fillna(1)

    logging.info(f"Data prepared: {len(df)} rows, Target distribution: {df['Target'].value_counts().to_dict()}")
    return df

# ---------------- Technical Indicators ---------------- #

def manual_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta>0,0).fillna(0)
    loss = (-delta.where(delta<0,0)).fillna(0)
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

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    horizons = [2,5,10,20,50]

    # Moving averages and ratios
    for h in horizons:
        if len(df) >= h:
            df[f'Close_MA_{h}'] = df['Close'].rolling(h, min_periods=1).mean()
            df[f'Close_Ratio_{h}'] = (df['Close'] / df[f'Close_MA_{h}']).fillna(1.0)
            df[f'Trend_{h}'] = df['Target'].shift(1).rolling(h, min_periods=1).sum().fillna(0) if 'Target' in df.columns else 0

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(20, min_periods=1).mean()
    rolling_std = df['Close'].rolling(20, min_periods=1).std(ddof=0)
    df['BB_Upper'] = rolling_mean + 2*rolling_std
    df['BB_Lower'] = rolling_mean - 2*rolling_std
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']).fillna(0)
    bb_width = df['BB_Width'].replace(0, np.nan)
    df['BB_Position'] = ((df['Close'] - df['BB_Lower']) / bb_width).fillna(0.5)

    # RSI & MACD
    df['RSI_14'] = manual_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = manual_macd(df['Close'])

    return df

def get_predictors(df):
    base = ['Close','Volume','Open','High','Low','Returns','High_Low_Ratio','Open_Close_Ratio']
    indicators = [c for c in df.columns if any(k in c for k in ['Close_Ratio_','Trend_','RSI','MACD','BB_','Volume_Ratio'])]
    predictors = [p for p in base+indicators if p in df.columns and (df[p].notna().sum()/len(df))>0.9]
    logging.info(f"Using {len(predictors)} predictors (example: {predictors[:5]})")
    return predictors

# ---------------- Modeling ---------------- #

def train_model(X, y):
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(random_state=1,n_jobs=-1))])
    param_grid = {'classifier__n_estimators':[100,200],'classifier__min_samples_split':[50,100],'classifier__max_depth':[10,20,None]}
    n_splits = min(5, len(X)//500)
    tscv = TimeSeriesSplit(n_splits=max(2,n_splits))
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='precision', n_jobs=-1, verbose=0)
    grid.fit(X,y)
    joblib.dump({'model':grid.best_estimator_, 'predictors': X.columns.tolist()}, MODEL_FILE)
    return grid.best_estimator_

# ---------------- Predictions ---------------- #

def predict_latest(model, data, predictors, n_days=5, threshold=THRESHOLD):
    if len(data) < n_days+100: return None
    train_size = len(data)-n_days
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    train_clean = train.dropna(subset=predictors+['Target'])
    test_clean = test.dropna(subset=predictors)
    if len(train_clean)<50 or len(test_clean)==0: return None
    model.fit(train_clean[predictors], train_clean['Target'])
    probs = model.predict_proba(test_clean[predictors])[:,1]
    preds = (probs>=threshold).astype(int)
    latest_df = pd.DataFrame({
        'Date': test_clean.index,
        'Close': test_clean['Close'].values,
        'Target': test.loc[test_clean.index,'Target'].values,
        'Prediction': preds,
        'Probability': probs
    })
    return latest_df
# -------------------- Plotting -------------------- #

def plot_save_close(df, filename='closing_price.png', ticker="^GSPC"):
    plt.figure(figsize=(14,7))
    df['Close'].plot(title=f'{ticker} Closing Price')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300, bbox_inches='tight')
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

# -------------------- Main -------------------- #

def main():
    logging.info("Starting Stock Predictor Pipeline...")
    df = fetch_stock_data()
    df = prepare_data(df)
    plot_save_close(df)
    df = add_technical_indicators(df)
    predictors = get_predictors(df)
    split_idx=int(len(df)*0.8)
    X_train=df[predictors].iloc[:split_idx].dropna()
    y_train=df['Target'].iloc[:split_idx].loc[X_train.index]
    X_test=df[predictors].iloc[split_idx:].dropna()
    y_test=df['Target'].iloc[split_idx:].loc[X_test.index]
    logging.info(f"Train/Test split: {len(X_train)}/{len(X_test)}")
    model=train_model(X_train,y_train)
    preds_backtest=backtest(df.iloc[:split_idx],model,predictors)
    plot_predictions(preds_backtest,"Train Backtest")
    if len(X_test)>0:
        test_probs=model.predict_proba(X_test)[:,1]
        test_preds=(test_probs>=THRESHOLD).astype(int)
        test_metrics=evaluate_model(y_test,test_preds,test_probs,"Test Set")
    else:
        test_metrics={}
    plot_feature_importance(model,predictors)
    latest=predict_latest(model,df,predictors,5)
    if latest is not None: plot_predictions(latest,"Latest 5 Days")
    logging.info("Pipeline completed successfully!")
    if test_metrics: logging.info(f"Test Precision: {test_metrics['Precision']:.4f}")
    print("\n" + "="*50)
    print("STOCK PREDICTOR COMPLETE!")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Plots saved in: {PLOT_DIR}/")
    print("="*50)

if __name__=="__main__":
    main()
