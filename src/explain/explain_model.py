"""
Model explainability using SHAP for credit card fraud detection.

This module provides functions to generate SHAP explanations
for model predictions.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Initialize SHAP
shap.initjs()

def explain_model_shap(
    model: Any,
    X: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100
) -> shap.Explanation:
    """
    Generate SHAP explanations for a model.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : np.ndarray
        Features to explain
    X_train : np.ndarray, optional
        Training data for background (if None, uses X)
    feature_names : List[str], optional
        Names of features
    max_samples : int
        Maximum number of samples to use for explanation
    
    Returns:
    --------
    shap.Explanation
        SHAP explanation object
    """
    # Limit samples for performance
    if len(X) > max_samples:
        X = X[:max_samples]
    
    # Use subset of training data as background
    if X_train is None:
        X_background = X[:min(100, len(X))]
    else:
        X_background = X_train[:min(100, len(X_train))]
    
    # Create appropriate SHAP explainer based on model type
    model_type = type(model).__name__.lower()
    
    if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
    else:
        # For linear models, use KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        shap_values = explainer.shap_values(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else None,
        data=X,
        feature_names=feature_names
    )
    
    return explanation

def plot_shap_summary(
    model: Any,
    X: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 100,
    save_path: Optional[str] = None
):
    """
    Generate and save SHAP summary plot.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : np.ndarray
        Features to explain
    X_train : np.ndarray, optional
        Training data for background
    feature_names : List[str], optional
        Names of features
    max_samples : int
        Maximum number of samples
    save_path : str, optional
        Path to save the plot
    """
    explanation = explain_model_shap(model, X, X_train, feature_names, max_samples)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(explanation.values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP summary plot saved to {save_path}")
    
    plt.close()

def get_local_explanation(
    model: Any,
    X_instance: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get local SHAP explanation for a single instance.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_instance : np.ndarray
        Single instance to explain (shape: (1, n_features) or (n_features,))
    X_train : np.ndarray, optional
        Training data for background
    feature_names : List[str], optional
        Names of features
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary with SHAP values and feature contributions
    """
    # Ensure 2D array
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    explanation = explain_model_shap(model, X_instance, X_train, feature_names, max_samples=1)
    
    shap_values = explanation.values[0] if hasattr(explanation, 'values') else explanation[0]
    
    # Create feature importance dictionary
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(shap_values))]
    
    feature_contributions = {
        name: float(value) for name, value in zip(feature_names, shap_values)
    }
    
    # Sort by absolute contribution
    sorted_contributions = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    return {
        'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
        'feature_contributions': feature_contributions,
        'top_features': dict(sorted_contributions[:10])  # Top 10 features
    }

def plot_shap_waterfall(
    model: Any,
    X_instance: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Generate and save SHAP waterfall plot for a single instance.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_instance : np.ndarray
        Single instance to explain
    X_train : np.ndarray, optional
        Training data for background
    feature_names : List[str], optional
        Names of features
    save_path : str, optional
        Path to save the plot
    """
    # Ensure 2D array
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    explanation = explain_model_shap(model, X_instance, X_train, feature_names, max_samples=1)
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation[0], show=False)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ SHAP waterfall plot saved to {save_path}")
    
    plt.close()

