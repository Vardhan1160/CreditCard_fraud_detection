"""
Unit tests for model prediction module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.models.predict import predict_fraud, predict_proba, predict_batch

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

@pytest.fixture
def trained_model(sample_data):
    """Create a trained model for testing."""
    X, y = sample_data
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    return model

def test_predict_fraud(trained_model, sample_data):
    """Test fraud prediction function."""
    X, _ = sample_data
    
    predictions = predict_fraud(trained_model, X, threshold=0.5)
    
    # Check output type and shape
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == X.shape[0]
    
    # Check values are binary
    assert np.all((predictions == 0) | (predictions == 1))

def test_predict_proba(trained_model, sample_data):
    """Test probability prediction function."""
    X, _ = sample_data
    
    probabilities = predict_proba(trained_model, X)
    
    # Check output type and shape
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape[0] == X.shape[0]
    
    # Check probabilities are in [0, 1] range
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

def test_predict_batch(trained_model, sample_data):
    """Test batch prediction function."""
    X, _ = sample_data
    
    results = predict_batch(trained_model, X, threshold=0.5, return_proba=True)
    
    # Check output structure
    assert 'predictions' in results
    assert 'probabilities' in results
    
    # Check types
    assert isinstance(results['predictions'], np.ndarray)
    assert isinstance(results['probabilities'], np.ndarray)
    
    # Check shapes
    assert results['predictions'].shape[0] == X.shape[0]
    assert results['probabilities'].shape[0] == X.shape[0]
    
    # Check probability range
    assert np.all(results['probabilities'] >= 0)
    assert np.all(results['probabilities'] <= 1)

def test_predict_with_dataframe(trained_model, sample_data):
    """Test prediction with DataFrame input."""
    X, _ = sample_data
    df = pd.DataFrame(X)
    
    predictions = predict_fraud(trained_model, df, threshold=0.5)
    probabilities = predict_proba(trained_model, df)
    
    # Check outputs
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(predictions) == len(df)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)

def test_predict_threshold(trained_model, sample_data):
    """Test prediction with different thresholds."""
    X, _ = sample_data
    
    # Low threshold (should predict more frauds)
    pred_low = predict_fraud(trained_model, X, threshold=0.1)
    
    # High threshold (should predict fewer frauds)
    pred_high = predict_fraud(trained_model, X, threshold=0.9)
    
    # Low threshold should predict more frauds than high threshold
    assert pred_low.sum() >= pred_high.sum()

