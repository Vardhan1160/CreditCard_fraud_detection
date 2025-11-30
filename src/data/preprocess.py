"""
Data preprocessing pipeline for credit card fraud detection.

This module provides preprocessing functions including scaling,
feature engineering, and handling missing values.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
from typing import Tuple, Optional

class CreditCardPreprocessor:
    """
    Preprocessor for credit card fraud detection data.
    
    Handles scaling of Time and Amount features, optional PCA,
    and saves/loads preprocessing artifacts.
    """
    
    def __init__(self, scale_type: str = 'robust', use_pca: bool = False, n_components: Optional[int] = None):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        scale_type : str
            Type of scaler: 'standard' or 'robust' (default: 'robust')
        use_pca : bool
            Whether to apply PCA transformation (default: False)
        n_components : int, optional
            Number of PCA components if use_pca=True
        """
        self.scale_type = scale_type
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame) -> 'CreditCardPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        
        Returns:
        --------
        self
        """
        self.feature_names_ = list(X.columns)
        
        # Select scaler
        if self.scale_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Scale Time and Amount
        scale_cols = ['Time', 'Amount']
        if all(col in X.columns for col in scale_cols):
            self.scaler = scaler.fit(X[scale_cols].values)
        else:
            # If columns don't exist, fit on all numeric columns
            self.scaler = scaler.fit(X.select_dtypes(include=[np.number]).values)
        
        # Optional PCA
        if self.use_pca:
            # First scale all features
            X_scaled = self.scaler.transform(X.select_dtypes(include=[np.number]).values)
            self.pca = PCA(n_components=self.n_components)
            self.pca.fit(X_scaled)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        
        Returns:
        --------
        np.ndarray
            Transformed features
        """
        if self.scaler is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Scale Time and Amount
        scale_cols = ['Time', 'Amount']
        if all(col in X.columns for col in scale_cols):
            X_scaled = X.copy()
            X_scaled[scale_cols] = self.scaler.transform(X[scale_cols].values)
        else:
            # Scale all numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_scaled = X.copy()
            X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols].values)
        
        # Convert to numpy array (preserve all features)
        X_array = X_scaled.select_dtypes(include=[np.number]).values
        
        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            X_array = self.pca.transform(X_array)
        
        return X_array
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def save(self, filepath: str):
        """
        Save preprocessor to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save preprocessor
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'CreditCardPreprocessor':
        """
        Load preprocessor from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to load preprocessor from
        
        Returns:
        --------
        CreditCardPreprocessor
            Loaded preprocessor
        """
        return joblib.load(filepath)

def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_type: str = 'robust',
    use_pca: bool = False,
    n_components: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, CreditCardPreprocessor]:
    """
    Preprocess training and test data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    scale_type : str
        Type of scaler: 'standard' or 'robust'
    use_pca : bool
        Whether to apply PCA
    n_components : int, optional
        Number of PCA components
    save_path : str, optional
        Path to save preprocessor
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, CreditCardPreprocessor]
        Transformed training data, transformed test data, and preprocessor
    """
    preprocessor = CreditCardPreprocessor(
        scale_type=scale_type,
        use_pca=use_pca,
        n_components=n_components
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    if save_path:
        preprocessor.save(save_path)
    
    return X_train_transformed, X_test_transformed, preprocessor

