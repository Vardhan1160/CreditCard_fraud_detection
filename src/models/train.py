"""
Model training pipeline for credit card fraud detection.

This module provides functions to train multiple models, perform
cross-validation, hyperparameter tuning, and save trained models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[str] = 'balanced',
    C: float = 1.0,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a logistic regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    class_weight : str, optional
        Class weight strategy ('balanced' or None)
    C : float
        Regularization strength
    random_state : int
        Random seed
    
    Returns:
    --------
    LogisticRegression
        Trained model
    """
    model = LogisticRegression(
        class_weight=class_weight,
        C=C,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    return model

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[str] = 'balanced',
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a random forest model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    class_weight : str, optional
        Class weight strategy
    n_estimators : int
        Number of trees
    max_depth : int, optional
        Maximum tree depth
    random_state : int
        Random seed
    
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    model = RandomForestClassifier(
        class_weight=class_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> xgb.XGBClassifier:
    """
    Train an XGBoost model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    scale_pos_weight : float, optional
        Weight for positive class (auto-calculated if None)
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    random_state : int
        Random seed
    
    Returns:
    --------
    xgb.XGBClassifier
        Trained model
    """
    if scale_pos_weight is None:
        # Calculate scale_pos_weight from class imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: Optional[float] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    scale_pos_weight : float, optional
        Weight for positive class
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Learning rate
    random_state : int
        Random seed
    
    Returns:
    --------
    lgb.LGBMClassifier
        Trained model
    """
    if scale_pos_weight is None:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.0017,
    random_state: int = 42
) -> IsolationForest:
    """
    Train an Isolation Forest (unsupervised anomaly detection).
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    contamination : float
        Expected proportion of anomalies
    random_state : int
        Random seed
    
    Returns:
    --------
    IsolationForest
        Trained model
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train)
    return model

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to training data.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    random_state : int
        Random seed
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Resampled features and labels
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'roc_auc',
    random_state: int = 42
) -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Parameters:
    -----------
    model : Any
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of folds
    scoring : str
        Scoring metric
    random_state : int
        Random seed
    
    Returns:
    --------
    Dict[str, float]
        CV scores (mean and std)
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    return {
        'mean': scores.mean(),
        'std': scores.std()
    }

def tune_hyperparameters(
    model: Any,
    param_distributions: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = 'roc_auc',
    random_state: int = 42
) -> Any:
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    Parameters:
    -----------
    model : Any
        Model to tune
    param_distributions : Dict
        Parameter distributions to search
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    n_iter : int
        Number of iterations
    cv : int
        Number of CV folds
    scoring : str
        Scoring metric
    random_state : int
        Random seed
    
    Returns:
    --------
    Any
        Best model from search
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_smote: bool = False,
    tune_hyperparams: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train all baseline models.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    use_smote : bool
        Whether to use SMOTE oversampling
    tune_hyperparams : bool
        Whether to tune hyperparameters
    random_state : int
        Random seed
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary of trained models
    """
    models = {}
    
    # Apply SMOTE if requested
    if use_smote:
        print("Applying SMOTE oversampling...")
        X_train_processed, y_train_processed = apply_smote(X_train, y_train, random_state)
    else:
        X_train_processed, y_train_processed = X_train, y_train
    
    # Logistic Regression
    print("Training Logistic Regression...")
    models['logistic'] = train_logistic_regression(
        X_train_processed, y_train_processed, random_state=random_state
    )
    
    # Random Forest
    print("Training Random Forest...")
    models['random_forest'] = train_random_forest(
        X_train_processed, y_train_processed, random_state=random_state
    )
    
    # XGBoost
    print("Training XGBoost...")
    models['xgboost'] = train_xgboost(
        X_train_processed, y_train_processed, random_state=random_state
    )
    
    # LightGBM
    print("Training LightGBM...")
    models['lightgbm'] = train_lightgbm(
        X_train_processed, y_train_processed, random_state=random_state
    )
    
    # Isolation Forest (unsupervised)
    print("Training Isolation Forest...")
    models['isolation_forest'] = train_isolation_forest(
        X_train, random_state=random_state
    )
    
    # Hyperparameter tuning for best model (XGBoost)
    if tune_hyperparams:
        print("Tuning XGBoost hyperparameters...")
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'scale_pos_weight': [None, (y_train == 0).sum() / (y_train == 1).sum()]
        }
        models['xgboost_tuned'] = tune_hyperparameters(
            xgb.XGBClassifier(random_state=random_state, use_label_encoder=False),
            param_dist,
            X_train_processed,
            y_train_processed,
            random_state=random_state
        )
    
    return models

def save_model(model: Any, filepath: str):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : Any
        Model to save
    filepath : str
        Path to save model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to load model from
    
    Returns:
    --------
    Any
        Loaded model
    """
    return joblib.load(filepath)

