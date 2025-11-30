"""
Main training script for credit card fraud detection.

This script orchestrates the entire training pipeline:
1. Load data
2. Preprocess
3. Train multiple models
4. Evaluate models
5. Save best model and artifacts
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data, split_features_target
from src.data.preprocess import preprocess_data
from src.models.train import train_all_models, save_model
from src.models.evaluate import evaluate_all_models, generate_evaluation_report
from src.explain.explain_model import plot_shap_summary

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Credit Card Fraud Detection - Training Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    try:
        df = load_data(sample_size=None)  # Use full dataset
    except FileNotFoundError:
        print("Dataset not found. Generating synthetic sample...")
        import subprocess
        subprocess.run([sys.executable, "scripts/sample_data.py"], check=True)
        df = load_data(sample_size=10000)  # Use smaller sample for demo
    
    # 2. Split features and target
    print("\n[2/6] Splitting features and target...")
    X, y = split_features_target(df)
    
    # 3. Train-test split
    print("\n[3/6] Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # 4. Preprocess
    print("\n[4/6] Preprocessing data...")
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    X_train_processed, X_test_processed, preprocessor = preprocess_data(
        X_train, X_test,
        scale_type='robust',
        use_pca=False,
        save_path=str(models_dir / 'preprocessor.pkl')
    )
    print(f"  Processed features shape: {X_train_processed.shape}")
    
    # 5. Train models
    print("\n[5/6] Training models...")
    models = train_all_models(
        X_train_processed, y_train,
        use_smote=False,  # Set to True to use SMOTE
        tune_hyperparams=False,  # Set to True for hyperparameter tuning
        random_state=42
    )
    
    # 6. Evaluate models
    print("\n[6/6] Evaluating models...")
    results_df = evaluate_all_models(models, X_test_processed, y_test)
    print("\nModel Performance Summary:")
    print(results_df.round(4))
    
    # Save results
    reports_dir = Path(__file__).parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    results_df.to_csv(reports_dir / 'model_comparison.csv')
    print(f"\n✓ Results saved to {reports_dir / 'model_comparison.csv'}")
    
    # Generate evaluation plots for best model
    best_model_name = results_df['roc_auc'].idxmax()
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f})")
    
    # Save best model
    save_model(best_model, str(models_dir / 'best_model.pkl'))
    
    # Generate evaluation plots
    figures_dir = reports_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    generate_evaluation_report(
        best_model, X_test_processed, y_test,
        model_name=best_model_name,
        output_dir=str(figures_dir)
    )
    
    # Generate SHAP summary plot
    print("\nGenerating SHAP explanations...")
    try:
        plot_shap_summary(
            best_model,
            X_test_processed[:100],  # Use subset for performance
            X_train=X_train_processed[:100],
            feature_names=list(X.columns),
            save_path=str(figures_dir / f'{best_model_name}_shap_summary.png')
        )
    except Exception as e:
        print(f"Warning: Could not generate SHAP plot: {e}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nSaved artifacts:")
    print(f"  - Model: {models_dir / 'best_model.pkl'}")
    print(f"  - Preprocessor: {models_dir / 'preprocessor.pkl'}")
    print(f"  - Results: {reports_dir / 'model_comparison.csv'}")
    print(f"  - Plots: {figures_dir}")

if __name__ == '__main__':
    main()

