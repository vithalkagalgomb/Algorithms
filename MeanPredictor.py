import numpy as np
from sklearn.metrics import mean_squared_error

# Sample regression data
y_train = np.array([3.2, 2.8, 3.5, 4.0, 3.9, 2.7, 3.1, 3.8])
y_test = np.array([3.0, 3.6, 3.3, 4.1])

# Mean Predictor: always predicts the mean of the training set
mean_value = y_train.mean()
predictions = np.full_like(y_test, fill_value=mean_value)

# Evaluate Mean Squared Error
mse = mean_squared_error(y_test, predictions)

print(f"Mean of training set: {mean_value:.2f}")
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test}")
print(f"Mean Squared Error: {mse:.2f}")
