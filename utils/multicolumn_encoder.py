from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator
from functools import partial

class MultiColumnEncoder(BaseEstimator):
    """Encoder for handling multiple categorical columns efficiently.
    
    This class provides functionality to encode multiple categorical columns
    while handling unknown values appropriately.
    """
    
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}
        oe = partial(OrdinalEncoder, handle_unknown='use_encoded_value', unknown_value=-1)
        
    def fit(self, X, y=None):
        """Fit the encoder on the training data.
        
        Args:
            X: Training data
            y: Target variable (not used, included for sklearn compatibility)
            
        Returns:
            self
        """
        # If no columns specified, use all object columns
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object']).columns
            
        # Fit an encoder for each column
        for col in self.columns:
            self.encoders[col] = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ).fit(X[[col]])
        return self
    
    def transform(self, X):
        """Transform the data using the fitted encoders.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X[[col]])
        return X_copy
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].inverse_transform(X[[col]])
        return X_copy