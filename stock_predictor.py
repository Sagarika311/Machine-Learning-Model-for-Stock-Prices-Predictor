# stock_predictor.py
# Machine Learning Model for Stock Prices Predictor
# Predicts if S&P 500 closing price will increase the next day.

import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Technical indicators
import ta  # For RSI, MACD, Bollinger Bands

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Data fetching function with error handling and recent data
def fetch_sp500_data(filename='sp500.csv', start_date='1950-01-01', end_date=datetime.now().strftime('%Y-%m-%d')):
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Loaded data from {filename}")
    else:
        try:
            ticker = yf.Ticker("^GSPC")
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data fetched. Check internet connection or ticker symbol.")
            df.to_csv(filename)
            print(f"Downloaded and saved data to {filename}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    return df

# Feature Engineering: Basic + Technical Indicators
def add_technical_indicators(df):
    # Basic features
    df['Returns'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Close'] / df['Open']
    
    # Moving averages and ratios
    horizons = [2, 5, 20, 60, 250]
    for horizon in horizons:
        df[f'Close_MA_{horizon}'] = df['Close'].rolling(horizon).mean()
        df[f'Close_Ratio_{horizon}'] = df['Close'] / df[f'Close_MA_{horizon}']
        df[f'Trend_{horizon}'] = df['Target'].shift(1).rolling(horizon).sum()
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    
    # Volume indicators
    df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
    
    return df

# Enhanced Predict Function with Probability Threshold
def predict(train, test, predictors, model, threshold=0.6):
    model.fit(train[predictors], train['Target'])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= threshold).astype(int)
    preds_series = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds_series, pd.Series(probs, index=test.index, name='Probs')], axis=1)
    return combined

# Enhanced Backtest with Time Series Split
def backtest(data, model, predictors, start=2500, step=250, threshold=0.6):
    all_predictions = []
    
    for i in range(start, len(data), step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:i+step].copy()
        predictions = predict(train, test, predictors, model, threshold)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

# Main execution
if __name__ == "__main__":
    # Step 1: Fetch and prepare data
    print("Fetching S&P 500 data...")
    sp500 = fetch_sp500_data()
    sp500.index = pd.to_datetime(sp500.index)
    
    # Clean data
    sp500.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')
    sp500['Tomorrow'] = sp500['Close'].shift(-1)
    sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)
    sp500 = sp500.loc['1990-01-01':].copy()
    sp500.dropna(subset=['Target'], inplace=True)
    
    print(f"Data shape after cleaning: {sp500.shape}")
    print(sp500.head())
    
    # Step 2: Plot closing price
    plt.figure(figsize=(14, 7))
    sp500['Close'].plot(title='S&P 500 Closing Price Over Time')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()
    
    # Step 3: Add technical indicators
    print("Adding technical indicators...")
    sp500 = add_technical_indicators(sp500)
    sp500.dropna(inplace=True)
    print(f"Data shape after features: {sp500.shape}")
    print(sp500.head())
    
    # Step 4: Define predictors
    predictors = ['Close', 'Volume', 'Open', 'High', 'Low', 'Returns', 'High_Low_Ratio', 'Open_Close_Ratio']
    predictors += [col for col in sp500.columns if any(ind in col for ind in ['Close_Ratio_', 'Trend_', 'RSI', 'MACD', 'BB_', 'Volume_Ratio'])]
    
    print(f"Number of predictors: {len(predictors)}")
    print("Sample predictors:", predictors[:10])  # Show first 10
    
    # Step 5: Model Pipeline and Hyperparameter Tuning
    print("Training model with hyperparameter tuning...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__min_samples_split': [50, 100],
        'classifier__max_depth': [10, 20, None]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='precision', n_jobs=-1, verbose=1)
    
    X = sp500[predictors]
    y = sp500['Target']
    grid_search.fit(X, y)
    
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV Precision: {grid_search.best_score_:.4f}")
    
    # Save model
    joblib.dump(model, 'stock_predictor_model.pkl')
    print("Model saved as 'stock_predictor_model.pkl'")
    
    # Step 6: Run backtest
    print("Running backtest...")
    predictions = backtest(sp500, model, predictors, threshold=0.6)
    print(predictions.head())
    
    # Step 7: Evaluation
    y_true = predictions['Target']
    y_pred = predictions['Predictions']
    y_prob = predictions['Probs']
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Step 8: Plot Predictions Over Time
    plt.figure(figsize=(14, 7))
    predictions[['Target', 'Predictions']].plot(title='Actual vs Predicted Targets Over Time')
    plt.ylabel('Target (1=Up, 0=Down)')
    plt.xlabel('Date')
    plt.legend(['Actual', 'Predicted'])
    plt.show()
    
    # Step 9: Feature Importance
    importances = model.named_steps['classifier'].feature_importances_
    feature_imp = pd.Series(importances, index=predictors).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 8))
    feature_imp.head(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.show()
    
    # Step 10: Prediction on Latest Data (last 5 days)
    if len(sp500) >= 5:
        latest_data = sp500.tail(5)
        train_data = sp500.iloc[:-5]
        latest_predictions = predict(train_data, latest_data, predictors, model, threshold=0.6)
        print("\nLatest Predictions (Last 5 Days):")
        print(latest_predictions)
    else:
        print("Not enough data for latest predictions.")
    
    print("\nScript completed successfully!")
