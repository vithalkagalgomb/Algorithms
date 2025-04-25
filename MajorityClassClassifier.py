import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

# Sample data: binary classification (0 or 1)
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y_train = np.array([0, 1, 1, 1, 0, 1, 1, 1])
X_test = np.array([[9], [10], [11], [12]])
y_test = np.array([1, 1, 0, 1])

# Majority Class Classifier: always predicts the most frequent class in training data
majority_class = Counter(y_train).most_common(1)[0][0]
predictions = np.full_like(y_test, fill_value=majority_class)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Majority Class: {majority_class}")
print(f"Predictions: {predictions}")
print(f"Actual:      {y_test}")
print(f"Accuracy: {accuracy:.2f}")
