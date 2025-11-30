"""
Streamlit web application for credit card fraud detection.

This module provides a user-friendly UI for making fraud predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.predict import predict_fraud, predict_proba
from src.data.preprocess import CreditCardPreprocessor
from src.explain.explain_model import get_local_explanation, plot_shap_summary
from src.data.load_data import load_data

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model and preprocessor
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'best_model.pkl'
PREPROCESSOR_PATH = Path(__file__).parent.parent / 'models' / 'preprocessor.pkl'

@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor (cached)."""
    model = None
    preprocessor = None
    
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if PREPROCESSOR_PATH.exists():
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    return model, preprocessor

def main():
    """Main Streamlit application."""
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("---")
    
    # Load artifacts
    model, preprocessor = load_model_and_preprocessor()
    
    if model is None or preprocessor is None:
        st.error("⚠️ Model or preprocessor not found. Please train the model first.")
        st.info("Run: `python -m src.models.train` to train the model.")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Single Prediction", "Batch Prediction", "Sample Dataset"])
    
    if page == "Single Prediction":
        single_prediction_page(model, preprocessor)
    elif page == "Batch Prediction":
        batch_prediction_page(model, preprocessor)
    elif page == "Sample Dataset":
        sample_dataset_page(model, preprocessor)

def single_prediction_page(model, preprocessor):
    """Single transaction prediction page."""
    st.header("Single Transaction Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Details")
        
        # Input fields
        time = st.number_input("Time", value=0.0, step=0.1)
        amount = st.number_input("Amount", value=0.0, step=0.01, min_value=0.0)
        
        # V features (collapsible)
        with st.expander("PCA Features (V1-V28)"):
            v_features = {}
            for i in range(1, 29):
                v_features[f'V{i}'] = st.number_input(
                    f"V{i}", value=0.0, step=0.01, key=f'v{i}'
                )
    
    with col2:
        st.subheader("Prediction")
        
        if st.button("Predict Fraud", type="primary"):
            try:
                # Create transaction dictionary
                transaction = {'Time': time, 'Amount': amount}
                transaction.update(v_features)
                df = pd.DataFrame([transaction])
                
                # Preprocess
                X = preprocessor.transform(df)
                
                # Predict
                prediction = predict_fraud(model, X, threshold=0.5)[0]
                probability = float(predict_proba(model, X)[0])
                
                # Display results
                if prediction == 1:
                    st.error(f"🚨 **FRAUD DETECTED**")
                    st.metric("Fraud Probability", f"{probability:.4f}")
                else:
                    st.success(f"✅ **NORMAL TRANSACTION**")
                    st.metric("Fraud Probability", f"{probability:.4f}")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if probability > 0.5 else "green"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP explanation
                if st.checkbox("Show SHAP Explanation"):
                    try:
                        explanation = get_local_explanation(model, X[0])
                        
                        st.subheader("Feature Contributions (SHAP)")
                        top_features = explanation.get('top_features', {})
                        
                        # Create bar chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=list(top_features.keys()),
                                y=list(top_features.values()),
                                marker_color=['red' if v > 0 else 'blue' for v in top_features.values()]
                            )
                        ])
                        fig.update_layout(
                            title="Top 10 Feature Contributions",
                            xaxis_title="Feature",
                            yaxis_title="SHAP Value",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate SHAP explanation: {e}")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")

def batch_prediction_page(model, preprocessor):
    """Batch prediction page."""
    st.header("Batch Transaction Prediction")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with transactions",
        type=['csv'],
        help="CSV should have columns: Time, V1-V28, Amount"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(df)} transactions")
            
            # Check required columns
            required_cols = ['Time', 'Amount'] + [f'V{i+1}' for i in range(28)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Display sample
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
                
                if st.button("Predict All", type="primary"):
                    with st.spinner("Processing predictions..."):
                        # Preprocess
                        X = preprocessor.transform(df[required_cols])
                        
                        # Predict
                        predictions = predict_fraud(model, X, threshold=0.5)
                        probabilities = predict_proba(model, X)
                        
                        # Add results to dataframe
                        df['Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities
                        df['Status'] = df['Prediction'].map({0: 'Normal', 1: 'Fraud'})
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df[['Time', 'Amount', 'Status', 'Fraud_Probability']])
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transactions", len(df))
                        with col2:
                            st.metric("Fraud Detected", int(df['Prediction'].sum()))
                        with col3:
                            st.metric("Fraud Rate", f"{df['Prediction'].mean()*100:.2f}%")
                        
                        # Distribution plot
                        fig = px.histogram(
                            df, x='Fraud_Probability', color='Status',
                            nbins=50, title="Fraud Probability Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv,
                            file_name="fraud_predictions.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

def sample_dataset_page(model, preprocessor):
    """Sample dataset inference page."""
    st.header("Sample Dataset Inference")
    
    st.info("This will run inference on a sample from the training dataset.")
    
    if st.button("Load Sample and Predict", type="primary"):
        try:
            with st.spinner("Loading sample data..."):
                # Load a small sample
                df = load_data(sample_size=100)
                X_df = df.drop(columns=['Class'])
                y_true = df['Class']
                
                # Preprocess
                X = preprocessor.transform(X_df)
                
                # Predict
                predictions = predict_fraud(model, X, threshold=0.5)
                probabilities = predict_proba(model, X)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Time': X_df['Time'].values,
                    'Amount': X_df['Amount'].values,
                    'True_Label': y_true.values,
                    'Prediction': predictions,
                    'Fraud_Probability': probabilities
                })
                results_df['Status'] = results_df['Prediction'].map({0: 'Normal', 1: 'Fraud'})
                results_df['True_Status'] = results_df['True_Label'].map({0: 'Normal', 1: 'Fraud'})
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df[['Time', 'Amount', 'True_Status', 'Status', 'Fraud_Probability']])
                
                # Calculate accuracy
                accuracy = (predictions == y_true.values).mean()
                fraud_detected = predictions.sum()
                actual_fraud = y_true.sum()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                with col2:
                    st.metric("Total Transactions", len(results_df))
                with col3:
                    st.metric("Fraud Detected", fraud_detected)
                with col4:
                    st.metric("Actual Fraud", int(actual_fraud))
                
                # Confusion matrix visualization
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, predictions)
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 16}
                ))
                fig.update_layout(title="Confusion Matrix", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure the dataset is available at data/raw/creditcard.csv")

if __name__ == "__main__":
    main()

