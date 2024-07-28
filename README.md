# Stock Market Prediction with Machine Learning in Python

The aim is to predict future stock market movements using historical S&P 500 data.

# Step 1: Importing the required modules
Import the following modules:

1. yfinance: Used to download financial data from Yahoo Finance
2. pandas: Used for data manipulation and analysis
3. os: For OS interaction, here to check if a file exists
4. sklearn.ensemble: To import RandomForestClassifier for prediction
5. sklearn.metrics: To import precision_score for model evaluation

# Step 2: Data Acquisition and Initial Processing
In this step, we check if the S&P 500 data exists locally. If not, we download it using yfinance.

Explanation:
- os.path.exists() checks if "sp500.csv" exists
- If it doesn't exist, yf.Ticker("^GSPC").history(period="max") downloads the entire history of S&P 500
- The data is then saved to a CSV file for future use
- pd.to_datetime() converts the index to datetime format

# Step 3: Data Visualization
Create a line plot of closing prices over time using sp500.plot.line(y="Close", use_index=True).

# Step 4: Data Preprocessing
Transform the raw data into a format suitable for machine learning:

- Remove unnecessary columns (Dividends and Stock Splits)
- Create a 'Tomorrow' column with the next day's closing price
- Create a 'Target' column (1 if price goes up, 0 if not)
- Filter data to start from 1990

# Step 5: Initial Model Creation
Create and train a Random Forest Classifier:

- Use RandomForestClassifier with 100 trees and a minimum of 100 samples to split an internal node
- Use the last 100 days as a test set, and the rest as training data
- Use "Close", "Volume", "Open", "High", "Low" as predictors

# Step 6: Model Evaluation
Evaluate the model's performance:

- Use model.predict() to make predictions on the test set
- Calculate the precision score using precision_score()

# Step 7: Backtesting Function
Create functions for prediction and backtesting:

- predict(): Fits the model on training data and makes predictions on test data
- backtest(): Implements a rolling window approach to test the model on different time periods

# Step 8: Feature Engineering
Create new features based on rolling averages and trends:

- Calculate price ratios and trends for different time horizons (2, 5, 60, 250, 1000 days)
- Add these new features to the dataset
- Remove any rows with NaN values

# Step 9: Improved Model
Create an improved Random Forest model:

- Increase the number of trees to 200
- Decrease the minimum samples to split to 50
- Update the predict() function to use a probability threshold of 0.6

# Step 10: Final Backtesting and Evaluation
Run the backtesting process with the new model and features:

- Use the backtest() function with the new predictors
- Calculate the distribution of predictions
- Calculate the precision score
- Calculate the actual market movement distribution

This code demonstrates a complete machine learning pipeline for stock market prediction, including data acquisition, preprocessing, model creation, feature engineering, and backtesting. The final model shows a slight improvement over baseline, but as with all financial predictions, it should be used cautiously.
