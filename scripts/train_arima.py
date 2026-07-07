import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the preprocessed clean closing prices
print("⏳ Loading cleaned asset prices...")
df = pd.read_csv('data/processed/cleaned_prices.csv', index_col='Date', parse_dates=True)
tsla_prices = df['TSLA'].asfreq('B').ffill()

# 2. Split data: 80% Training history, 20% Evaluation Test set
split_idx = int(len(tsla_prices) * 0.8)
train, test = tsla_prices.iloc[:split_idx], tsla_prices.iloc[split_idx:]

# 3. Fit an ARIMA(1,1,1) model (Differenced d=1 to handle the non-stationarity found in Task 1)
print("🤖 Training ARIMA(1,1,1) model on TSLA...")
model = ARIMA(train, order=(1, 1, 1))
model_fitted = model.fit()

# 4. Forecast across the length of our testing period
print("🔮 Generating out-of-sample forecasts...")
forecast = model_fitted.forecast(steps=len(test))
forecast.index = test.index

# 5. Evaluate the forecasting precision metrics
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print("\n📊 --- TSLA ARIMA Forecasting Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
