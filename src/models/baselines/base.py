"""
Base class for baseline models
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import StandardScaler


class BaselineModel(ABC):
    """
    Abstract base class for all baseline models
    Provides unified interface for training and evaluation
    """
    
    def __init__(self, name="BaselineModel"):
        self.name = name
        self.is_fitted = False
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    @abstractmethod
    def fit(self, X_train, y_train, **kwargs):
        """
        Train the model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,) or (n_samples, n_traits)
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def predict(self, X_test, **kwargs):
        """
        Make predictions
        
        Args:
            X_test: Test features (n_samples, n_features)
            **kwargs: Additional arguments
        
        Returns:
            Predictions (n_samples,) or (n_samples, n_traits)
        """
        pass
    
    def fit_predict(self, X_train, y_train, X_test, **kwargs):
        """Fit and predict in one call"""
        self.fit(X_train, y_train, **kwargs)
        return self.predict(X_test, **kwargs)
    
    def preprocess(self, X, y=None, fit=False):
        """
        Preprocess features and targets
        
        Args:
            X: Features
            y: Targets (optional)
            fit: Whether to fit scalers
        
        Returns:
            Preprocessed X and y
        """
        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            if y is not None:
                if y.ndim == 1:
                    y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                else:
                    y_scaled = self.scaler_y.fit_transform(y)
            else:
                y_scaled = None
        else:
            X_scaled = self.scaler_X.transform(X)
            if y is not None:
                if y.ndim == 1:
                    y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
                else:
                    y_scaled = self.scaler_y.transform(y)
            else:
                y_scaled = None
        
        return X_scaled, y_scaled
    
    def inverse_transform_predictions(self, y_pred):
        """Inverse transform predictions back to original scale"""
        if y_pred.ndim == 1:
            return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        else:
            return self.scaler_y.inverse_transform(y_pred)
    
    def get_params(self):
        """Get model parameters"""
        return {}
    
    def set_params(self, **params):
        """Set model parameters"""
        for key, value in params.items():
            setattr(self, key, value)
