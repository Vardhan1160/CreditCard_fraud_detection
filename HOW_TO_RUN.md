# How to Run the Credit Card Fraud Detection Project

## Step-by-Step Guide

### Step 1: Setup Python Environment

Open PowerShell in the project directory and run:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

This will install all necessary libraries (pandas, scikit-learn, xgboost, fastapi, streamlit, etc.)

### Step 3: Prepare the Dataset

**Option A: Use Real Kaggle Dataset (Recommended)**
1. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in `data\raw\creditcard.csv`

**Option B: Generate Synthetic Sample Data**
```powershell
python scripts/sample_data.py
```

This creates a small synthetic dataset for testing/demo purposes.

### Step 4: Train the Model

```powershell
python scripts/train_model.py
```

**What this does:**
- Loads the dataset
- Preprocesses the data
- Trains multiple models (Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest)
- Evaluates all models
- Saves the best model to `models\best_model.pkl`
- Saves preprocessor to `models\preprocessor.pkl`
- Generates evaluation plots in `reports\figures\`

**Expected output:**
- Model performance comparison table
- Saved model files
- Evaluation plots (confusion matrix, ROC curve, PR curve, SHAP plots)

**Note:** Training may take 5-15 minutes depending on your system and dataset size.

### Step 5: Start the API (FastAPI)

Open a **new PowerShell window** (keep the training one open if you want to see results), activate venv, and run:

```powershell
# Activate venv in new window
.\venv\Scripts\Activate.ps1

# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**API will be available at:**
- Main API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Step 6: Run the Streamlit Web App (Optional)

Open **another PowerShell window**, activate venv, and run:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Start Streamlit app
streamlit run webapp/app_streamlit.py
```

**Web app will open at:** http://localhost:8501

## Quick Test

### Test the API

Once the API is running, you can test it:

**Using PowerShell:**
```powershell
# Test health endpoint
Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET

# Test prediction (example)
$body = @{
    transaction = @{
        Time = 0.0
        V1 = -1.359807134
        V2 = -0.072781173
        V3 = 2.536346738
        V4 = 1.378155224
        V5 = -0.33826177
        V6 = 0.462387778
        V7 = 0.239598554
        V8 = 0.098697901
        V9 = 0.36378697
        V10 = 0.090794172
        V11 = -0.551599533
        V12 = -0.617800856
        V13 = -0.991389847
        V14 = -0.311169354
        V15 = 1.468176972
        V16 = -0.470400525
        V17 = 0.207971242
        V18 = 0.02579058
        V19 = 0.40399296
        V20 = 0.251412098
        V21 = -0.018306778
        V22 = 0.277837576
        V23 = -0.11047391
        V24 = 0.066928075
        V25 = 0.128539358
        V26 = -0.189114844
        V27 = 0.133558377
        V28 = -0.021053053
        Amount = 149.62
    }
    include_explanation = $false
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

**Or use the interactive docs:** Visit http://localhost:8000/docs and use the web interface to test endpoints.

### Test the Streamlit App

1. Open http://localhost:8501 in your browser
2. Use the "Single Prediction" page to enter transaction details
3. Or use "Batch Prediction" to upload a CSV file
4. Or use "Sample Dataset" to test on sample data

## Running Tests

```powershell
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## Docker Deployment (Alternative)

If you prefer using Docker:

```powershell
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Troubleshooting

### Issue: "Model not found" error when starting API

**Solution:** Make sure you've trained the model first:
```powershell
python scripts/train_model.py
```

### Issue: "Dataset not found" error

**Solution:** Either:
1. Download the real dataset from Kaggle and place it at `data\raw\creditcard.csv`
2. Or generate synthetic data: `python scripts/sample_data.py`

### Issue: Port already in use

**Solution:** Change the port:
```powershell
# For API
uvicorn src.api.app:app --host 0.0.0.0 --port 8001

# For Streamlit
streamlit run webapp/app_streamlit.py --server.port 8502
```

### Issue: Execution policy error for venv activation

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## What Gets Created

After running the training script, you'll have:

- `models\best_model.pkl` - The best performing model
- `models\preprocessor.pkl` - The fitted preprocessor
- `reports\model_comparison.csv` - Performance metrics for all models
- `reports\figures\` - Evaluation plots:
  - Confusion matrices
  - ROC curves
  - Precision-Recall curves
  - SHAP summary plots

## Next Steps

1. **Explore the notebook:** Open `notebooks\eda_and_modeling.ipynb` in Jupyter
2. **Modify training:** Edit `scripts\train_model.py` to change hyperparameters
3. **Customize API:** Edit `src\api\app.py` to add new endpoints
4. **Enhance web app:** Edit `webapp\app_streamlit.py` to add features

## Summary

**Minimum steps to get started:**
1. `python -m venv venv` → `.\venv\Scripts\Activate.ps1`
2. `pip install -r requirements.txt`
3. `python scripts/sample_data.py` (or download real dataset)
4. `python scripts/train_model.py`
5. `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` (in new terminal)
6. Visit http://localhost:8000/docs to test the API

That's it! Your fraud detection system is now running.

