# portfolio_report.py
import os
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
from stock_predictor import (
    fetch_stock_data,
    add_technical_indicators,
    plot_feature_importance,
    plot_closing_price,
    predict_latest,
    evaluate_model,
    PLOT_DIR,
    THRESHOLD
)

# Settings
MODEL_FILE = 'stock_predictor_model.pkl'
TICKER = "^GSPC"
N_DAYS = 5
PDF_FILE = os.path.join(PLOT_DIR, f"{TICKER}_portfolio_report.pdf")

os.makedirs(PLOT_DIR, exist_ok=True)

# 1️⃣ Load model bundle
bundle = joblib.load(MODEL_FILE)
model = bundle['model']
predictors = bundle['predictors']
print(f"Loaded model with {len(predictors)} predictors")

# 2️⃣ Fetch latest stock data
df = fetch_stock_data(TICKER)

# 3️⃣ Add technical indicators
df = add_technical_indicators(df)

# 4️⃣ Generate plots
closing_plot_path = os.path.join(PLOT_DIR, f'{TICKER}_closing_price.png')
plot_closing_price(df, ticker=TICKER)

feature_plot_path = os.path.join(PLOT_DIR, f'{TICKER}_feature_importance.png')
plot_feature_importance(model, predictors)

# 5️⃣ Latest N-day predictions
latest_df = predict_latest(model, df, predictors, n_days=N_DAYS)

latest_plot_path = os.path.join(PLOT_DIR, f'{TICKER}_latest_predictions.png')
if latest_df is not None:
    plt.figure(figsize=(10,6))
    plt.plot(latest_df['Date'], latest_df['Close'], marker='o', label='Close')
    plt.scatter(latest_df['Date'], latest_df['Close'], c=latest_df['Predictions'], cmap='RdYlGn', label='Predicted Up/Down')
    plt.title(f'{TICKER} - Latest {N_DAYS} Predictions')
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(latest_plot_path, dpi=300)
    plt.close()

# 6️⃣ Evaluate model on the last 20% as test
split_idx = int(len(df) * 0.8)
X_test = df[predictors].iloc[split_idx:].dropna()
y_test = df['Target'].iloc[split_idx:].loc[X_test.index]

if len(X_test) > 0:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)
    metrics = evaluate_model(y_test, y_pred, y_prob, "Test Set")
else:
    metrics = {}

# 7️⃣ Generate PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, f"{TICKER} Stock Predictor Report", ln=True, align='C')

pdf.set_font("Arial", '', 12)
pdf.ln(5)
pdf.multi_cell(0, 8, "This report shows historical closing prices, feature importance, "
                       f"and predictions for the latest {N_DAYS} trading days using a RandomForest model.")

# Add plots
for img_path, title in [(closing_plot_path, "Closing Prices"), 
                        (feature_plot_path, "Feature Importance"), 
                        (latest_plot_path, f"Latest {N_DAYS}-Day Predictions")]:
    if os.path.exists(img_path):
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, title, ln=True, align='C')
        pdf.image(img_path, x=15, y=30, w=180)
        
# Add metrics table
if metrics:
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Test Set Metrics", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    for key, val in metrics.items():
        pdf.cell(0, 8, f"{key}: {val:.4f}", ln=True)

# 8️⃣ Save PDF
pdf.output(PDF_FILE)
print(f"Portfolio PDF report saved: {PDF_FILE}")
