# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib

from stock_predictor import (
    fetch_stock_data,
    prepare_data,
    add_technical_indicators,
    get_predictors,
    train_model,
    predict_latest,
    MODEL_FILE,
    PLOT_DIR
)

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 S&P 500 Stock Predictor")

# ---------------- Sidebar ---------------- #
st.sidebar.header("Options")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1990-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
n_latest_days = st.sidebar.slider("Number of latest days to predict", 1, 10, 5)

# ---------------- Fetch & Prepare Data ---------------- #
with st.spinner("Fetching data..."):
    df = fetch_stock_data(start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"))
    df = prepare_data(df)
    df = add_technical_indicators(df)

st.success(f"Data ready: {len(df)} rows")
st.dataframe(df.tail(5))

# ---------------- Load or Train Model ---------------- #
with st.spinner("Loading or training model..."):
    if os.path.exists(MODEL_FILE):
        model_bundle = joblib.load(MODEL_FILE)
        # Handle old vs new model format
        if isinstance(model_bundle, dict) and 'model' in model_bundle and 'predictors' in model_bundle:
            model = model_bundle['model']
            predictors = model_bundle['predictors']
        else:
            model = model_bundle  # old format (direct pipeline/classifier)
            predictors = get_predictors(df)
        st.success("Model loaded successfully.")
    else:
        predictors = get_predictors(df)
        st.warning("Model not found. Training new model...")
        model = train_model(df[predictors], df['Target'])
        st.success("Model trained and saved.")

# ---------------- Predictions ---------------- #
if st.button("Predict Latest Days"):
    latest_df = predict_latest(model, df, predictors, n_days=n_latest_days)
    if latest_df is not None and not latest_df.empty:
        st.subheader(f"Predictions for Last {n_latest_days} Days")
        st.dataframe(latest_df.style.format({"Close": "{:.2f}", "Probs": "{:.2f}"}))

        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(latest_df['Date'], latest_df['Close'], label='Close Price', color='blue')
        if 'Predictions' in latest_df.columns:
            scatter = ax.scatter(latest_df['Date'], latest_df['Close'],
                                 c=latest_df['Predictions'], cmap='coolwarm', s=100, label='Prediction')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.set_title("Close Price and Predictions")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough data to predict.")

# ---------------- Historical Price ---------------- #
if st.checkbox("Show Historical Closing Price"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close Price', color='green')
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("S&P 500 Closing Price")
    st.pyplot(fig)
