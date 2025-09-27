# stock_predictor.py
# Machine Learning Model for Stock Prices Predictor
# Predicts if closing price will increase the next day using S&P 500 data.
# Author: Assistant (Updated for robustness and deployment)

import os
import logging
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import yfinance as yf

# Technical indicators using pandas_ta (pure Python)
import pandas_ta as ta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configuration
TICKER = "^GSPC"  # S&P 500
START_DATE = '1990-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_FILE = 'sp500.csv'
MODEL_FILE = 'stock_predictor_model.pkl'
PLOT_DIR = 'plots'
THRESHOLD = 0.6  # Probability threshold for predictions

# Create plot directory
os.makedirs(PLOT_DIR, exist_ok=True)

def fetch_stock_data(ticker=TICKER, filename=DATA_FILE, start=START_DATE, end=END_DATE):
    """Fetch and cache stock data using yfinance."""
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        logging.info(f"Loaded data from {filename}")
    else:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end)
            if df.empty:
                raise ValueError(f"No data fetched for {ticker}. Check connection or dates.")
            df.to_csv(filename)
            logging.info(f"Downloaded and saved data for {ticker} to {filename}")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise
    return df

def prepare_data(df):
    """Clean and prepare data: create target and basic features."""
    # Clean columns
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')
    
    # Create target: 1 if next close > current close
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    
    # Slice from start date and drop NaNs
    df = df.loc[START_DATE:].copy()
    df.dropna(subset=['Target'], inplace=True)
    
    if len(df) < 1000:
        raise ValueError("Insufficient data after cleaning. Need at least 1000 rows.")
    
    logging.info(f"Data prepared: {len(df)} rows")
    return df

def add_technical_indicators(df):
    """Add advanced technical indicators using pandas_ta."""
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Close'] / df['Open']
    
    # Moving averages and ratios/trends
    horizons = [2, 5, 20, 60, 250]
    for h in horizons:
        df[f'Close_MA_{h}'] = df['Close'].rolling(h).mean()
        df[f'Close_Ratio_{h}'] = df['Close'] / df[f'Close_MA_{h}']
        df[f'Trend_{h}'] = df['Target'].shift(1).rolling(h).sum()
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=20)
    df['BB_High'] = bb['BBU_20_2.0']
    df['BB_Low'] = bb['BBL_20_2.0']
    df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    
    # Volume ratio
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    # Drop any new NaNs
    df.dropna(inplace=True)
    
    logging.info(f"Technical indicators added: {len(df)} rows remaining")
    return df

def get_predictors(df):
    """Define list of predictor columns."""
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low', 'Returns', 'High_Low_Ratio', 'Open_Close_Ratio']
    indicators = [col for col in df.columns if any(ind in col for ind in ['Close_Ratio_', 'Trend_', 'RSI', 'MACD', 'BB_', 'Volume_Ratio'])]
    predictors.extend(indicators)
    return [p for p in predictors if p in df.columns]  # Ensure all exist

def train_model(X, y):
    """Train model with pipeline and hyperparameter tuning using time series CV."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1, n_jobs=-1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__min_samples_split': [50, 100],
        'classifier__max_depth': [10, 20, None]
    }
    
    # Time series split for CV
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='precision', n_jobs=-1, verbose=1)
    
    grid_search.fit(X, y)
    
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best CV Precision: {grid_search.best_score_:.4f}")
    
    # Save model
    joblib.dump(grid_search.best_estimator_, MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")
    
    return grid_search.best_estimator_

def backtest(data, model, predictors, start=2500, step=250, threshold=THRESHOLD):
    """Rolling window backtest."""
    all_preds = []
    max_idx = len(data)
    
    for i in range(start, max_idx, step):
        end_i = min(i + step, max_idx)
        train = data.iloc[:i]
        test = data.iloc[i:end_i]
        
        if len(train) < 100 or len(test) < 10:  # Skip if insufficient data
            continue
        
        model.fit(train[predictors], train['Target'])
        probs = model.predict_proba(test[predictors])[:, 1]
        preds = (probs >= threshold).astype(int)
        
        pred_df = pd.DataFrame({
            'Target': test['Target'],
            'Predictions': preds,
            'Probs': probs
        }, index=test.index)
        
        all_preds.append(pred_df)
    
    if not all_preds:
        raise ValueError("Backtest failed: No valid windows found.")
    
    return pd.concat(all_preds)

def evaluate_model(y_true, y_pred, y_prob):
    """Compute and print evaluation metrics."""
    print("\n=== Model Evaluation ===")
    print(classification_report(y_true, y_pred))
    
    metrics = {
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }
    
    for name, score in metrics.items():
        print(f"{name}: {score:.4f}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics

def plot_predictions(predictions):
    """Plot actual vs predicted over time."""
    plt.figure(figsize=(14, 7))
    predictions[['Target', 'Predictions']].plot(title='Actual vs Predicted Targets Over Time')
    plt.ylabel('Target (1=Up, 0=Down)')
    plt.xlabel('Date')
    plt.legend(['Actual', 'Predicted'])
    plt.savefig(os.path.join(PLOT_DIR, 'predictions_over_time.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, predictors):
    """Plot top feature importances."""
    try:
        importances = model.named_steps['classifier'].feature_importances_
    except:
        importances = model.feature_importances_
    
    feature_imp = pd.Series(importances, index=predictors).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    feature_imp.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()

def predict_latest(model, data, predictors, n_days=5, threshold=THRESHOLD):
    """Predict on the latest n_days of data."""
    if len(data) < n_days + 100:
        logging.warning("Insufficient data for latest predictions.")
        return None
    
    train = data.iloc[-(n_days + 100):-n_days]
    test = data.tail(n_days)
    
    model.fit(train[predictors], train['Target'])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    
    latest_df = pd.DataFrame({
        'Target': test['Target'],
        'Predictions': preds,
        'Probs': probs,
        'Close': test['Close']
    }, index=test.index)
    
    logging.info("Latest Predictions (Last 5 Days):")
    print(latest_df)
    return latest_df

def plot_closing_price(df):
    """Plot historical closing price."""
    plt.figure(figsize=(14, 7))
    df['Close'].plot(title=f'{TICKER} Closing Price Over Time')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.savefig(os.path.join(PLOT_DIR, 'closing_price.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    try:
        # Step 1: Fetch data
        logging.info("Starting stock predictor pipeline...")
        df = fetch_stock_data()
        
        # Step 2: Prepare data
        df = prepare_data(df)
        plot_closing_price(df)
        
        # Step 3: Add features
        df = add_technical_indicators(df)
        
        # Step 4: Get predictors
        predictors = get_predictors(df)
        logging.info(f"Using {len(predictors)} predictors")
        
        # Step 5: Prepare X, y and split for final eval
        X = df[predictors]
        y = df['Target']
        
        # Time-based split (80% train, 20% test)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Step 6: Train model
        model = train_model(X_train, y_train)
        
        # Step 7: Backtest on full data (excluding last test set for fairness)
        backtest_data = df.iloc[:split_idx]
        predictions = backtest(backtest_data, model, predictors)
        
        # Step 8: Evaluate on held-out test set
        test_predictions = predict(pd.concat([backtest_data, X_test]), pd.DataFrame(X_test, index=X_test.index).join(y_test), predictors, model)
        metrics = evaluate_model(test_predictions['Target'], test_predictions['Predictions'], test_predictions['Probs'])
        
        # Step 9: Plots
        plot_predictions(predictions)
        plot_feature_importance(model, predictors)
        
        # Step 10: Latest predictions
        predict_latest(model, df, predictors)
        
        logging.info("Pipeline completed successfully!")
        logging.info(f"Plots saved to '{PLOT_DIR}' folder.")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
