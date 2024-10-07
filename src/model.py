import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add bias term
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate coefficients
        identity = np.eye(X_bias.shape[1])
        identity[0, 0] = 0  # Don't regularize the bias term
        self.coefficients = np.linalg.inv(X_bias.T.dot(X_bias) + self.alpha * identity).dot(X_bias.T).dot(y)
        
        # Extract intercept
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept