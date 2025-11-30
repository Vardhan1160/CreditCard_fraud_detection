"""
Model evaluation utilities for credit card fraud detection.

This module provides functions to evaluate models, compute metrics,
and generate evaluation plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from pathlib import Path
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a model and return comprehensive metrics.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    
    Returns:
    --------
    Dict[str, float]
        Dictionary of evaluation metrics
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    elif hasattr(model, 'decision_function'):
        # For Isolation Forest
        y_pred = model.predict(X_test)
        y_pred = (y_pred == -1).astype(int)  # Convert -1/1 to 0/1
        y_pred_proba = model.decision_function(X_test)
        # Normalize decision function to [0, 1] range
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
        y_pred_proba = 1 - y_pred_proba  # Invert so higher = more anomalous
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = np.zeros_like(y_pred, dtype=float)
    
    # Calculate metrics
    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    # Add confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
    
    return metrics

def evaluate_all_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison DataFrame.
    
    Parameters:
    -----------
    models : Dict[str, Any]
        Dictionary of trained models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for all models
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    df = df.set_index('model')
    
    return df

def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot and save confusion matrix.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    elif hasattr(model, 'decision_function'):
        y_pred = model.predict(X_test)
        y_pred = (y_pred == -1).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot and save ROC curve.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
        y_pred_proba = 1 - y_pred_proba
    else:
        return  # Can't plot ROC without probabilities
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.close()

def plot_precision_recall_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
):
    """
    Plot and save precision-recall curve.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_pred_proba = model.decision_function(X_test)
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
        y_pred_proba = 1 - y_pred_proba
    else:
        return
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to {save_path}")
    
    plt.close()

def generate_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Generate comprehensive evaluation report with plots.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    output_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    Dict[str, float]
        Evaluation metrics
    """
    metrics = evaluate_model(model, X_test, y_test, model_name)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plots
        plot_confusion_matrix(
            model, X_test, y_test, model_name,
            save_path=output_dir / f'{model_name}_confusion_matrix.png'
        )
        plot_roc_curve(
            model, X_test, y_test, model_name,
            save_path=output_dir / f'{model_name}_roc_curve.png'
        )
        plot_precision_recall_curve(
            model, X_test, y_test, model_name,
            save_path=output_dir / f'{model_name}_pr_curve.png'
        )
    
    return metrics

