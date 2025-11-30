# Credit Card Fraud Detection - Quick Start

## 🚀 Quick Start Guide

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Generate sample data (if dataset not available)
python scripts/sample_data.py
```

### 2. Train Model

```bash
python scripts/train_model.py
```

### 3. Start API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4. Run Streamlit App

```bash
streamlit run webapp/app_streamlit.py
```

## 📍 Key Locations

- **Models**: `models/best_model.pkl`
- **Preprocessor**: `models/preprocessor.pkl`
- **Results**: `reports/model_comparison.csv`
- **Plots**: `reports/figures/`

## 🔗 Endpoints

- API: http://localhost:8000
- Streamlit: http://localhost:8501
- API Docs: http://localhost:8000/docs

## 📊 Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"transaction": {"Time": 0.0, "V1": -1.36, ..., "Amount": 149.62}}'
```

See `README.md` for full documentation.

