"""
GBLUP (Genomic Best Linear Unbiased Prediction) baseline
Standard statistical method for genomic prediction
"""

import numpy as np
from scipy import linalg
from .base import BaselineModel


class GBLUP(BaselineModel):
    """
    GBLUP implementation
    
    Uses genomic relationship matrix (GRM) to predict breeding values
    
    Args:
        lambda_reg: Regularization parameter (default: 1.0)
    """
    
    def __init__(self, lambda_reg=1.0):
        super().__init__(name="GBLUP")
        self.lambda_reg = lambda_reg
        self.alpha = None
        self.X_train = None
    
    def compute_grm(self, X):
        """
        Compute Genomic Relationship Matrix (GRM)
        
        GRM = XX^T / p
        where p is the number of markers
        
        Args:
            X: Genotype matrix (n_samples, n_markers)
        
        Returns:
            GRM (n_samples, n_samples)
        """
        n_samples, n_markers = X.shape
        
        # Center genotypes
        X_centered = X - X.mean(axis=0)
        
        # Compute GRM
        GRM = (X_centered @ X_centered.T) / n_markers
        
        return GRM
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Fit GBLUP model
        
        Solves: (K + λI)α = y
        where K is the GRM
        
        Args:
            X_train: Training genotypes (n_train, n_markers)
            y_train: Training phenotypes (n_train,) or (n_train, n_traits)
        """
        # Preprocess
        X_train_scaled, y_train_scaled = self.preprocess(X_train, y_train, fit=True)
        
        # Store training data
        self.X_train = X_train_scaled
        
        # Compute GRM
        K = self.compute_grm(X_train_scaled)
        
        # Add regularization
        K_reg = K + self.lambda_reg * np.eye(K.shape[0])
        
        # Solve for alpha
        # α = (K + λI)^(-1) y
        try:
            self.alpha = linalg.solve(K_reg, y_train_scaled, assume_a='pos')
        except:
            # Fallback to lstsq if solve fails
            self.alpha = linalg.lstsq(K_reg, y_train_scaled)[0]
        
        self.is_fitted = True
    
    def predict(self, X_test, **kwargs):
        """
        Predict using GBLUP
        
        Prediction: ŷ = K_test,train α
        where K_test,train is the relationship between test and train samples
        
        Args:
            X_test: Test genotypes (n_test, n_markers)
        
        Returns:
            Predictions (n_test,) or (n_test, n_traits)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Preprocess test data
        X_test_scaled, _ = self.preprocess(X_test, fit=False)
        
        # Compute relationship between test and train
        n_test = X_test_scaled.shape[0]
        n_train = self.X_train.shape[0]
        n_markers = X_test_scaled.shape[1]
        
        # Center test genotypes using training mean
        X_test_centered = X_test_scaled - self.X_train.mean(axis=0)
        X_train_centered = self.X_train - self.X_train.mean(axis=0)
        
        # K_test,train = X_test X_train^T / p
        K_test_train = (X_test_centered @ X_train_centered.T) / n_markers
        
        # Predict
        y_pred_scaled = K_test_train @ self.alpha
        
        # Inverse transform
        y_pred = self.inverse_transform_predictions(y_pred_scaled)
        
        return y_pred
    
    def get_params(self):
        """Get model parameters"""
        return {'lambda_reg': self.lambda_reg}
    
    def set_params(self, **params):
        """Set model parameters"""
        if 'lambda_reg' in params:
            self.lambda_reg = params['lambda_reg']


def test_gblup():
    """Test GBLUP implementation"""
    print("Testing GBLUP...")
    
    # Generate dummy data
    np.random.seed(42)
    n_train = 100
    n_test = 20
    n_markers = 500
    
    X_train = np.random.randint(0, 3, (n_train, n_markers))
    X_test = np.random.randint(0, 3, (n_test, n_markers))
    
    # Generate phenotypes with some genetic signal
    true_effects = np.random.randn(n_markers) * 0.1
    y_train = X_train @ true_effects + np.random.randn(n_train) * 0.5
    y_test = X_test @ true_effects + np.random.randn(n_test) * 0.5
    
    # Fit GBLUP
    model = GBLUP(lambda_reg=1.0)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Compute correlation
    from scipy.stats import pearsonr
    corr, _ = pearsonr(y_test, y_pred)
    
    print(f"Training samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Markers: {n_markers}")
    print(f"Test correlation: {corr:.4f}")
    
    # Test multi-trait
    y_train_multi = np.column_stack([y_train, y_train * 0.8])
    y_test_multi = np.column_stack([y_test, y_test * 0.8])
    
    model_multi = GBLUP(lambda_reg=1.0)
    model_multi.fit(X_train, y_train_multi)
    y_pred_multi = model_multi.predict(X_test)
    
    print(f"\nMulti-trait predictions shape: {y_pred_multi.shape}")
    
    print("\nGBLUP test passed!")


if __name__ == '__main__':
    test_gblup()
