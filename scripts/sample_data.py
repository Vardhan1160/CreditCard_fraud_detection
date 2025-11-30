"""
Generate synthetic credit card fraud detection dataset for demo purposes.

This script creates a small synthetic dataset that mimics the structure
of the Kaggle Credit Card Fraud Detection dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.0017, random_state=42):
    """
    Generate synthetic credit card transaction data.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    fraud_ratio : float
        Ratio of fraud cases (default ~0.17% to match real dataset)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with same structure as creditcard.csv
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Generate Time feature (seconds elapsed)
    time_normal = np.random.exponential(scale=1000, size=n_normal)
    time_fraud = np.random.exponential(scale=500, size=n_fraud)
    time = np.concatenate([time_normal, time_fraud])
    time = np.cumsum(time)
    
    # Generate Amount feature
    amount_normal = np.random.lognormal(mean=3, sigma=1.5, size=n_normal)
    amount_fraud = np.random.lognormal(mean=4, sigma=2, size=n_fraud)
    amount = np.concatenate([amount_normal, amount_fraud])
    
    # Generate V1-V28 PCA features (simplified)
    # In real dataset, these are PCA-transformed, so we'll create correlated features
    data = {}
    
    # Base features for normal transactions
    base_normal = np.random.randn(n_normal, 28)
    base_fraud = np.random.randn(n_fraud, 28)
    
    # Add some structure to fraud cases
    base_fraud[:, :5] += np.random.randn(n_fraud, 5) * 2  # Make some features more extreme
    
    for i in range(28):
        data[f'V{i+1}'] = np.concatenate([base_normal[:, i], base_fraud[:, i]])
    
    # Add Time and Amount
    data['Time'] = time
    data['Amount'] = amount
    
    # Create target variable
    class_labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    df = pd.DataFrame(data)
    df['Class'] = class_labels
    df = df.iloc[indices].reset_index(drop=True)
    
    # Reorder columns to match original format
    cols = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount', 'Class']
    df = df[cols]
    
    return df

def main():
    """Generate and save synthetic dataset."""
    output_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'creditcard.csv'
    
    print("Generating synthetic credit card fraud detection dataset...")
    df = generate_synthetic_data(n_samples=10000, fraud_ratio=0.0017)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Synthetic dataset saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    print(f"  Normal cases: {(df['Class']==0).sum()} ({(df['Class']==0).mean()*100:.2f}%)")

if __name__ == '__main__':
    main()

