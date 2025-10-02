# 📈 Stock Price Direction Predictor

A **machine learning project** that predicts whether the **next day’s closing price** of a stock/index will go **Up or Down**.
Built using **Python, scikit-learn, yFinance, and technical indicators**, with an interactive **Streamlit app** for easy exploration.

---

## 🔑 Features

* Fetches real historical data from **Yahoo Finance** (fallback to dummy data if offline).
* Implements technical indicators **manually** (RSI, MACD, Bollinger Bands, moving averages, etc.).
* Handles **time-series cross-validation** (avoiding lookahead bias).
* Trains a **Random Forest Classifier** with grid search for hyperparameter tuning.
* Produces detailed **evaluation metrics** (precision, recall, F1, ROC-AUC, confusion matrix).
* Backtests predictions over historical data.
* Interactive **Streamlit app** for trying different tickers.

---

## 📂 Project Structure

```
stock-predictor/
├── stock_predictor.py       # Main ML pipeline
├── app.py                   # Streamlit web app
├── requirements.txt         # Dependencies
├── sp500.csv                # Cached S&P 500 data (auto-downloaded)
├── stock_predictor_model.pkl # Saved trained model
├── plots/                   # Generated plots (confusion matrix, feature importance, etc.)
└── README.md
```

---

## ⚙️ Installation

Clone the repo and install requirements:

```bash
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run the pipeline script

```bash
python stock_predictor.py
```

* Downloads stock data (default: **S&P 500**).
* Trains model, evaluates, backtests.
* Saves plots in `/plots`.
* Saves trained model as `stock_predictor_model.pkl`.

Example output (Test Set Evaluation):

```
Precision: 0.68
Recall:    0.54
F1-Score:  0.60
Accuracy:  0.62
ROC-AUC:   0.65
```

### 2. Launch the Streamlit app

```bash
streamlit run app.py
```

Try different tickers (e.g., AAPL, TSLA) and see predictions for the latest days:

📸 **Screenshot Placeholder**:
*(Insert Streamlit UI screenshot here — table of predictions + closing price chart)*

---

## 📊 Example Outputs

Confusion Matrix:
📸 *[insert confusion matrix plot here]*

Feature Importance:
📸 *[insert feature importance plot here]*

Closing Price Trend:
📸 *[insert closing price chart here]*

---

## 🚀 Deployment

You can deploy the app for free using **Streamlit Cloud**:

* Push this repo to GitHub.
* Go to [share.streamlit.io](https://share.streamlit.io/), connect your repo, and deploy.

Alternative: Host with **Render** or **Heroku** if you prefer Flask/FastAPI APIs.

---

## 🔮 Future Improvements

* Add support for **multiple ML models** (XGBoost, CatBoost, LSTMs).
* Extend backtest with **portfolio/trading strategy simulation**.
* Add more features (fundamental indicators, sentiment analysis).
* Deploy a **live daily predictor** (cron job + auto email/Telegram alerts).

---

## 📌 Disclaimer

⚠️ This project is for **educational purposes only**.
It is **not financial advice**. Predictions are experimental and should not be used for real trading decisions.


