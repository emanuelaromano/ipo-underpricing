import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies log transformation to positive values.
    Handles zero and negative values by adding a small constant before transformation.
    """
    
    def __init__(self, add_constant=1.0):
        self.add_constant = add_constant
        
    def fit(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return np.log(X + self.add_constant)
    
    def fit_transform(self, X, y=None):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        return np.log(X + self.add_constant)
    
    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return np.exp(X) - self.add_constant

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features)