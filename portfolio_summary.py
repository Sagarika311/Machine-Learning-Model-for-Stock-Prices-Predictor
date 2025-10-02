# portfolio_summary.py
import os
import joblib
import matplotlib.pyplot as plt
from stock_predictor import (
    fetch_stock_data,
    add_technical_indicators,
    plot_predictions,
    plot_feature_importance,
    plot_closing_price,
    predict_latest,
    PLOT_DIR
)

# Settings
MODEL_FILE = 'stock_predictor_model.pkl'
TICKER = "^GSPC"
N_DAYS = 5

# Ensure plot dir
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

# 4️⃣ Plot historical closing price
plot_closing_price(df, ticker=TICKER)

# 5️⃣ Plot feature importance
plot_feature_importance(model, predictors)

# 6️⃣ Predict latest N_DAYS
latest_df = predict_latest(model, df, predictors, n_days=N_DAYS)

# 7️⃣ Plot latest predictions
if latest_df is not None:
    plot_predictions(latest_df, f"{TICKER} - Latest {N_DAYS} Days")

# 8️⃣ Optionally save summary figure combining plots
fig_summary_path = os.path.join(PLOT_DIR, f"{TICKER}_summary.png")
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Closing price
df['Close'].plot(ax=axes[0], title=f'{TICKER} Closing Price')
axes[0].set_ylabel("Price ($)")

# Feature importance
importances = model.named_steps['classifier'].feature_importances_
axes[1].barh(predictors, importances)
axes[1].set_title("Feature Importances")

# Latest predictions
if latest_df is not None:
    axes[2].plot(latest_df['Date'], latest_df['Close'], marker='o', label='Close')
    axes[2].scatter(latest_df['Date'], latest_df['Close'], c=latest_df['Predictions'], cmap='RdYlGn', label='Predicted Up/Down')
    axes[2].set_title(f'Latest {N_DAYS} Predictions')
    axes[2].set_ylabel("Price ($)")
    axes[2].legend()

plt.tight_layout()
plt.savefig(fig_summary_path, dpi=300)
plt.show()
print(f"Portfolio summary saved: {fig_summary_path}")
