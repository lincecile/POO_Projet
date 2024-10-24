import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

class LinearRegression(Model):
    def _init_(self, intercept: bool):
        self.intercept = intercept

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
    
    def predict(self, X: np.ndarray):
        if self.intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y_pred = X @ self.beta
        return y_pred

