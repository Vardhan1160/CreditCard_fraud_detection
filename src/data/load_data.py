"""
Data loading utilities for credit card fraud detection.

This module provides functions to load and sample the credit card
fraud detection dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_data(data_path: Optional[str] = None, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load the credit card fraud detection dataset.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the CSV file. If None, uses default path.
    sample_size : int, optional
        If provided, randomly sample this many rows for faster processing.
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    if data_path is None:
        # Default path
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'data' / 'raw' / 'creditcard.csv'
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please download from Kaggle or run 'python scripts/sample_data.py' "
            f"to generate synthetic data."
        )
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} rows from {len(df)} total rows...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"  Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
    
    return df

def split_features_target(df: pd.DataFrame, target_col: str = 'Class') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataset into features and target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Features and target
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def get_feature_names(df: pd.DataFrame, target_col: str = 'Class') -> list:
    """
    Get list of feature names (excluding target).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    
    Returns:
    --------
    list
        List of feature names
    """
    return [col for col in df.columns if col != target_col]

