"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import CreditCardPreprocessor

def test_preprocessor_initialization():
    """Test preprocessor initialization."""
    preprocessor = CreditCardPreprocessor(scale_type='robust', use_pca=False)
    assert preprocessor.scale_type == 'robust'
    assert preprocessor.use_pca == False

def test_preprocessor_fit_transform():
    """Test preprocessor fit and transform."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    data = {
        'Time': np.random.randn(n_samples) * 1000,
        'Amount': np.random.lognormal(3, 1, n_samples),
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
    }
    df = pd.DataFrame(data)
    
    # Fit and transform
    preprocessor = CreditCardPreprocessor(scale_type='robust', use_pca=False)
    X_transformed = preprocessor.fit_transform(df)
    
    # Check output shape
    assert X_transformed.shape[0] == n_samples
    assert X_transformed.shape[1] == len(df.columns)
    
    # Check no NaNs
    assert not np.isnan(X_transformed).any()

def test_preprocessor_save_load(tmp_path):
    """Test preprocessor save and load."""
    # Create sample data
    np.random.seed(42)
    data = {
        'Time': np.random.randn(50) * 1000,
        'Amount': np.random.lognormal(3, 1, 50),
    }
    df = pd.DataFrame(data)
    
    # Fit preprocessor
    preprocessor = CreditCardPreprocessor(scale_type='robust', use_pca=False)
    preprocessor.fit(df)
    
    # Save
    save_path = tmp_path / 'preprocessor.pkl'
    preprocessor.save(str(save_path))
    
    # Load
    loaded_preprocessor = CreditCardPreprocessor.load(str(save_path))
    
    # Check they produce same output
    original_output = preprocessor.transform(df)
    loaded_output = loaded_preprocessor.transform(df)
    
    np.testing.assert_array_almost_equal(original_output, loaded_output)

def test_preprocessor_with_pca():
    """Test preprocessor with PCA."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    data = {
        'Time': np.random.randn(n_samples) * 1000,
        'Amount': np.random.lognormal(3, 1, n_samples),
        'V1': np.random.randn(n_samples),
        'V2': np.random.randn(n_samples),
    }
    df = pd.DataFrame(data)
    
    # Fit and transform with PCA
    preprocessor = CreditCardPreprocessor(scale_type='robust', use_pca=True, n_components=2)
    X_transformed = preprocessor.fit_transform(df)
    
    # Check output shape (should be reduced by PCA)
    assert X_transformed.shape[0] == n_samples
    assert X_transformed.shape[1] == 2  # n_components
    
    # Check no NaNs
    assert not np.isnan(X_transformed).any()

"""