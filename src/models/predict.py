"""
Prediction utilities for credit card fraud detection.

This module provides functions to make predictions using trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
import joblib
from pathlib import Path

def predict_fraud(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Predict fraud using a trained model.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : Union[np.ndarray, pd.DataFrame]
        Features to predict on
    threshold : float
        Classification threshold (default: 0.5)
    
    Returns:
    --------
    np.ndarray
        Binary predictions (0 = Normal, 1 = Fraud)
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    elif hasattr(model, 'decision_function'):
        # For Isolation Forest
        predictions = model.predict(X)
        predictions = (predictions == -1).astype(int)
    else:
        predictions = model.predict(X)
    
    return predictions

def predict_proba(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame]
) -> np.ndarray:
    """
    Get fraud probabilities from a trained model.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : Union[np.ndarray, pd.DataFrame]
        Features to predict on
    
    Returns:
    --------
    np.ndarray
        Fraud probabilities
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]
    elif hasattr(model, 'decision_function'):
        # For Isolation Forest, convert decision function to probability-like score
        scores = model.decision_function(X)
        # Normalize to [0, 1] range
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        probabilities = 1 - scores  # Invert so higher = more anomalous
    else:
        # Fallback: return predictions as probabilities
        predictions = model.predict(X)
        probabilities = predictions.astype(float)
    
    return probabilities

def predict_batch(
    model: Any,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5,
    return_proba: bool = True
) -> Dict[str, np.ndarray]:
    """
    Make batch predictions with probabilities.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : Union[np.ndarray, pd.DataFrame]
        Features to predict on
    threshold : float
        Classification threshold
    return_proba : bool
        Whether to return probabilities
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with 'predictions' and optionally 'probabilities'
    """
    predictions = predict_fraud(model, X, threshold)
    
    result = {'predictions': predictions}
    
    if return_proba:
        probabilities = predict_proba(model, X)
        result['probabilities'] = probabilities
    
    return result

def load_model_and_predict(
    model_path: str,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Load a model and make predictions.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    X : Union[np.ndarray, pd.DataFrame]
        Features to predict on
    threshold : float
        Classification threshold
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Predictions and probabilities
    """
    model = joblib.load(model_path)
    return predict_batch(model, X, threshold, return_proba=True)

