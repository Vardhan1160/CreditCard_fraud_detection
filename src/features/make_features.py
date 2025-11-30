"""
Feature engineering utilities for credit card fraud detection.

This module provides functions to create additional features
from the raw transaction data.
"""

import pandas as pd
import numpy as np
from typing import Optional

def create_time_features(df: pd.DataFrame, time_col: str = 'Time') -> pd.DataFrame:
    """
    Create time-based features from Time column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Name of time column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional time features
    """
    df = df.copy()
    
    if time_col in df.columns:
        # Hour of day (assuming Time is in seconds)
        df['hour'] = (df[time_col] / 3600) % 24
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (assuming 7-day cycle)
        df['day_of_week'] = (df[time_col] / (3600 * 24)) % 7
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_amount_features(df: pd.DataFrame, amount_col: str = 'Amount') -> pd.DataFrame:
    """
    Create amount-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    amount_col : str
        Name of amount column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional amount features
    """
    df = df.copy()
    
    if amount_col in df.columns:
        # Log transform (handle zeros)
        df['amount_log'] = np.log1p(df[amount_col])
        
        # Binned amount
        df['amount_binned'] = pd.cut(
            df[amount_col],
            bins=[0, 10, 50, 100, 500, 1000, float('inf')],
            labels=[0, 1, 2, 3, 4, 5]
        ).astype(float)
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between V features and Amount/Time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with interaction features
    """
    df = df.copy()
    
    # Interaction between Amount and first few V features
    if 'Amount' in df.columns:
        v_cols = [col for col in df.columns if col.startswith('V')]
        if v_cols:
            # Use first 5 V features for interactions
            for v_col in v_cols[:5]:
                df[f'{v_col}_amount'] = df[v_col] * df['Amount']
    
    return df

def engineer_features(
    df: pd.DataFrame,
    include_time: bool = True,
    include_amount: bool = True,
    include_interactions: bool = False
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    include_time : bool
        Whether to create time features
    include_amount : bool
        Whether to create amount features
    include_interactions : bool
        Whether to create interaction features
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    if include_time:
        df = create_time_features(df)
    
    if include_amount:
        df = create_amount_features(df)
    
    if include_interactions:
        df = create_interaction_features(df)
    
    return df

