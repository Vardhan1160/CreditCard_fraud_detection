"""
FastAPI application for credit card fraud detection predictions.

This module provides a REST API endpoint for making fraud predictions.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.predict import predict_fraud, predict_proba
from src.data.preprocess import CreditCardPreprocessor
from src.explain.explain_model import get_local_explanation

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessor on startup
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'best_model.pkl'
PREPROCESSOR_PATH = Path(__file__).parent.parent.parent / 'models' / 'preprocessor.pkl'

model = None
preprocessor = None
feature_names = None

@app.on_event("startup")
async def load_artifacts():
    """Load model and preprocessor on application startup."""
    global model, preprocessor, feature_names
    
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠ Warning: Model not found at {MODEL_PATH}")
        
        if PREPROCESSOR_PATH.exists():
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"✓ Preprocessor loaded from {PREPROCESSOR_PATH}")
        else:
            print(f"⚠ Warning: Preprocessor not found at {PREPROCESSOR_PATH}")
        
        # Default feature names (V1-V28, Time, Amount)
        feature_names = ['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount']
        
    except Exception as e:
        print(f"⚠ Error loading artifacts: {e}")

class Transaction(BaseModel):
    """Single transaction data model."""
    Time: float = Field(..., description="Time in seconds")
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Amount: float = Field(..., description="Transaction amount")

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    transaction: Transaction
    include_explanation: bool = Field(default=False, description="Include SHAP explanation")

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Prediction (0=Normal, 1=Fraud)")
    probability: float = Field(..., description="Fraud probability")
    explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP explanation if requested")

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    transactions: List[Transaction]
    include_explanations: bool = Field(default=False, description="Include SHAP explanations")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[int]
    probabilities: List[float]
    explanations: Optional[List[Dict[str, Any]]] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single transaction prediction",
            "/predict/batch": "POST - Batch transaction predictions",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud for a single transaction.
    
    Parameters:
    -----------
    request : PredictionRequest
        Transaction data and options
    
    Returns:
    --------
    PredictionResponse
        Prediction, probability, and optional explanation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = request.transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        prediction = predict_fraud(model, X, threshold=0.5)[0]
        probability = float(predict_proba(model, X)[0])
        
        # Get explanation if requested
        explanation = None
        if request.include_explanation:
            try:
                explanation = get_local_explanation(
                    model, X[0], feature_names=feature_names
                )
            except Exception as e:
                print(f"Warning: Could not generate explanation: {e}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for multiple transactions.
    
    Parameters:
    -----------
    request : BatchPredictionRequest
        List of transactions and options
    
    Returns:
    --------
    BatchPredictionResponse
        Predictions, probabilities, and optional explanations
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")
    
    try:
        # Convert transactions to DataFrame
        transactions_dict = [t.dict() for t in request.transactions]
        df = pd.DataFrame(transactions_dict)
        
        # Preprocess
        X = preprocessor.transform(df)
        
        # Predict
        predictions = predict_fraud(model, X, threshold=0.5).tolist()
        probabilities = predict_proba(model, X).tolist()
        
        # Get explanations if requested
        explanations = None
        if request.include_explanations:
            try:
                explanations = []
                for i in range(len(X)):
                    exp = get_local_explanation(
                        model, X[i], feature_names=feature_names
                    )
                    explanations.append(exp)
            except Exception as e:
                print(f"Warning: Could not generate explanations: {e}")
        
        return BatchPredictionResponse(
            predictions=[int(p) for p in predictions],
            probabilities=[float(prob) for prob in probabilities],
            explanations=explanations
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

