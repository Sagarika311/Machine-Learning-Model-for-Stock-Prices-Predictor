# 📈 Stock Price Direction Predictor

**Predict the next day’s stock price movement** (Up/Down) using historical data and machine learning.
Built with **Python**, **scikit-learn**, **yFinance**, and **technical indicators**, featuring an **interactive Streamlit app** for exploration and visualization.

---

## 🔑 Key Features

* Fetches historical stock data from **Yahoo Finance** or cached CSV.
* Computes **technical indicators manually** (RSI, MACD, Bollinger Bands, moving averages, etc.).
* Applies **time-series aware training** (avoiding lookahead bias).
* Trains a **Random Forest Classifier** with **grid search hyperparameter tuning**.
* Provides **evaluation metrics**: precision, recall, F1-score, ROC-AUC, and confusion matrix.
* Supports **backtesting** over historical data to validate performance.
* Interactive **Streamlit app** for experimenting with tickers and prediction horizons.

---

## 📂 Project Structure

```
stock-predictor/
├── stock_predictor.py        # Main ML pipeline
├── predict_new.py            # Predict recent/new data
├── app.py                    # Streamlit web app
├── requirements.txt          # Python dependencies
├── sp500.csv                 # Cached S&P 500 data
├── stock_predictor_model.pkl # Saved trained model
├── plots/                    # Generated plots (feature importance, confusion matrix, etc.)
├── screenshots/              # Streamlit UI screenshots
└── README.md
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Sagarika311/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run the ML pipeline

```bash
python stock_predictor.py
```

This will:

* Fetch historical stock data (default: **S&P 500**) or use cached CSV.
* Train the model and evaluate its performance.
* Backtest predictions on historical data.
* Save plots in `/plots`.
* Save the trained model as `stock_predictor_model.pkl`.

**Sample Test Set Metrics:**

```
Precision: 0.68
Recall:    0.54
F1-Score:  0.60
Accuracy:  0.62
ROC-AUC:   0.65
```

---

### 2. Quick Predictions on Recent Data

Use `predict_new.py` to generate predictions for the **last N days** without running the full pipeline:

```bash
python predict_new.py
```

* Default: predicts for the **last 5 trading days**.
* Outputs a table with:

  * Date
  * Closing Price
  * Predicted Direction (Up/Down)
  * Probability of an Up day

**Example Output:**

```
 Date       Close  Predictions  Probability_Up
2025-09-26  4500        1           0.78
2025-09-27  4525        0           0.43
2025-09-28  4510        1           0.65
2025-09-29  4528        1           0.81
2025-09-30  4540        0           0.55
```

*Note: Model must exist (`stock_predictor_model.pkl`). Otherwise, run `stock_predictor.py` first.*

---

### 3. Launch the interactive Streamlit app

```bash
streamlit run app.py
```

📸 **Streamlit UI Screenshot**:
![Streamlit UI](screenshots/streamlit_ui(1).png)
![Streamlit UI](screenshots/streamlit_ui(2).png)
*(Displays table of predictions + closing price chart)*
![Streamlit UI](screenshots/streamlit_ui(3).png)
![Streamlit UI](screenshots/streamlit_ui(4).png)

---

## 📊 Example Outputs

**Confusion Matrix**
![Confusion Matrix](plots/confusion_matrix_test_set.png)

**Feature Importance**
![Feature Importance](plots/feature_importance.png)

**Closing Price Trend**
![Closing Price Trend](plots/closing_price.png)

---

## 🚀 Deployment

Deploy easily using **Streamlit Cloud**:

1. Push your repo to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/), connect your repo, and deploy.

Alternative hosting: **Render** or **Heroku** for Flask/FastAPI APIs.

---

## 🔮 Future Improvements

* Add support for additional ML models (XGBoost, CatBoost, LSTM).
* Enhance backtesting with **portfolio/trading strategy simulation**.
* Integrate **fundamental analysis** and **sentiment indicators**.
* Deploy a **live daily predictor** (cron job with email/Telegram alerts).

---

## 📌 Disclaimer

⚠️ **For educational purposes only.**
This project **does not constitute financial advice**. Predictions are experimental and **should not be used for real trading decisions**.

---

## 👩‍💻 Author

**Sagarika Bhagat**
GitHub: [Sagarika311](https://github.com/Sagarika311)
