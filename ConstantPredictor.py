import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# Download Tesla stock data
symbol = 'TSLA'
data = yf.download(symbol, start='2024-01-01', end='2024-04-01')

# Use closing prices
prices = data['Close']

# Constant Predictor: always predicts the mean of the training set
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
test_prices = prices[train_size:]

constant_value = train_prices.mean()
predictions = np.full_like(test_prices, fill_value=constant_value)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(prices.index[train_size:], test_prices, label='Actual')
plt.plot(prices.index[train_size:], predictions, label='Constant Prediction', linestyle='--')
plt.title('Constant Predictor Demo on Tesla (TSLA) Stock')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Print evaluation metric (e.g., Mean Squared Error)
mse = np.mean((test_prices - predictions) ** 2)
print(f"Mean Squared Error of Constant Predictor: {mse:.2f}")
