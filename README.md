# Credit Card Fraud Detection

A complete, production-ready machine learning system for detecting fraudulent credit card transactions. This project includes data preprocessing, multiple model implementations, hyperparameter tuning, model evaluation, explainability (SHAP), a REST API (FastAPI), and a web interface (Streamlit).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Running the API](#running-the-api)
  - [Using the Streamlit Web App](#using-the-streamlit-web-app)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Limitations and Future Work](#limitations-and-future-work)

## 🎯 Project Overview

This project implements a comprehensive fraud detection system that:

- **Preprocesses** credit card transaction data with robust scaling and optional feature engineering
- **Trains multiple models** including Logistic Regression, Random Forest, XGBoost, LightGBM, and Isolation Forest
- **Handles class imbalance** using SMOTE, ADASYN, and class weights
- **Evaluates models** with comprehensive metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- **Explains predictions** using SHAP values for model interpretability
- **Serves predictions** via a FastAPI REST API
- **Provides a web interface** using Streamlit for easy interaction

## ✨ Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Isolation Forest
- **Class Imbalance Handling**: SMOTE, ADASYN, class weights
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal model performance
- **Model Explainability**: SHAP values for feature importance and local explanations
- **REST API**: FastAPI-based API with single and batch prediction endpoints
- **Web Interface**: Streamlit app for interactive predictions
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **CI/CD**: GitHub Actions workflow for automated testing
- **Comprehensive Testing**: Unit tests for key modules

## 📊 Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

### Download Instructions

1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Download `creditcard.csv`
3. Place it in `data/raw/creditcard.csv`

**Note**: If you don't have access to the Kaggle dataset, you can generate a synthetic sample:

```bash
python scripts/sample_data.py
```

### Dataset Details

- **Size**: ~284,807 transactions
- **Features**: 30 features (28 PCA-transformed + Time + Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Class Imbalance**: Highly imbalanced (~0.172% fraud cases)

See `data/README.md` for more details.

## 🚀 Installation

### Prerequisites

- Python 3.9 or 3.10
- pip or conda

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Prepare the dataset**:

```bash
# Option 1: Download from Kaggle and place at data/raw/creditcard.csv
# Option 2: Generate synthetic sample
python scripts/sample_data.py
```

## ⚡ Quick Start

### 1. Train the Model

```bash
python scripts/train_model.py
```

This will:
- Load and preprocess the data
- Train multiple models
- Evaluate and compare models
- Save the best model to `models/best_model.pkl`
- Generate evaluation plots in `reports/figures/`

### 2. Start the API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

### 3. Run the Streamlit App

```bash
streamlit run webapp/app_streamlit.py
```

App will be available at: http://localhost:8501

## 📁 Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/
│   │   └── creditcard.csv          # Dataset (not included, see data/README.md)
│   └── README.md                    # Dataset instructions
├── notebooks/
│   └── eda_and_modeling.ipynb      # EDA and modeling experiments
├── src/
│   ├── data/
│   │   ├── load_data.py            # Data loading utilities
│   │   └── preprocess.py           # Preprocessing pipeline
│   ├── features/
│   │   └── make_features.py        # Feature engineering
│   ├── models/
│   │   ├── train.py                # Model training
│   │   ├── predict.py               # Prediction functions
│   │   └── evaluate.py              # Model evaluation
│   ├── explain/
│   │   └── explain_model.py        # SHAP explanations
│   └── api/
│       └── app.py                  # FastAPI application
├── webapp/
│   └── app_streamlit.py            # Streamlit web app
├── scripts/
│   ├── train_model.py              # Main training script
│   ├── run_training.sh             # Training shell script
│   └── sample_data.py              # Synthetic data generator
├── tests/
│   ├── test_preprocess.py          # Preprocessing tests
│   └── test_model_predict.py       # Prediction tests
├── models/                         # Saved models (created after training)
├── reports/
│   └── figures/                    # Evaluation plots (created after training)
├── Dockerfile                      # Docker image for API
├── Dockerfile.streamlit            # Docker image for Streamlit
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI/CD workflow
└── README.md                       # This file
```

## 📖 Usage

### Training Models

Run the complete training pipeline:

```bash
python scripts/train_model.py
```

**Options** (edit `scripts/train_model.py`):
- `use_smote=True`: Enable SMOTE oversampling
- `tune_hyperparams=True`: Enable hyperparameter tuning
- `sample_size=None`: Use full dataset (or specify number for faster training)

**Output**:
- Trained models saved to `models/`
- Evaluation metrics saved to `reports/model_comparison.csv`
- Plots saved to `reports/figures/`

### Running the API

#### Local Development

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Production

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `POST /predict`: Single transaction prediction
- `POST /predict/batch`: Batch transaction predictions

### Using the Streamlit Web App

```bash
streamlit run webapp/app_streamlit.py
```

The app provides three pages:
1. **Single Prediction**: Enter transaction details manually
2. **Batch Prediction**: Upload CSV file with transactions
3. **Sample Dataset**: Run inference on sample data

### Example API Requests

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction": {
      "Time": 0.0,
      "V1": -1.359807134,
      "V2": -0.072781173,
      "V3": 2.536346738,
      "V4": 1.378155224,
      "V5": -0.33826177,
      "V6": 0.462387778,
      "V7": 0.239598554,
      "V8": 0.098697901,
      "V9": 0.36378697,
      "V10": 0.090794172,
      "V11": -0.551599533,
      "V12": -0.617800856,
      "V13": -0.991389847,
      "V14": -0.311169354,
      "V15": 1.468176972,
      "V16": -0.470400525,
      "V17": 0.207971242,
      "V18": 0.02579058,
      "V19": 0.40399296,
      "V20": 0.251412098,
      "V21": -0.018306778,
      "V22": 0.277837576,
      "V23": -0.11047391,
      "V24": 0.066928075,
      "V25": 0.128539358,
      "V26": -0.189114844,
      "V27": 0.133558377,
      "V28": -0.021053053,
      "Amount": 149.62
    },
    "include_explanation": true
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "Time": 0.0,
        "V1": -1.359807134,
        "V2": -0.072781173,
        "V3": 2.536346738,
        "V4": 1.378155224,
        "V5": -0.33826177,
        "V6": 0.462387778,
        "V7": 0.239598554,
        "V8": 0.098697901,
        "V9": 0.36378697,
        "V10": 0.090794172,
        "V11": -0.551599533,
        "V12": -0.617800856,
        "V13": -0.991389847,
        "V14": -0.311169354,
        "V15": 1.468176972,
        "V16": -0.470400525,
        "V17": 0.207971242,
        "V18": 0.02579058,
        "V19": 0.40399296,
        "V20": 0.251412098,
        "V21": -0.018306778,
        "V22": 0.277837576,
        "V23": -0.11047391,
        "V24": 0.066928075,
        "V25": 0.128539358,
        "V26": -0.189114844,
        "V27": 0.133558377,
        "V28": -0.021053053,
        "Amount": 149.62
      }
    ],
    "include_explanations": false
  }'
```

## 📈 Model Performance

After training, model performance is saved to `reports/model_comparison.csv`. Typical results on the test set:

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| Logistic Regression | ~0.85 | ~0.60 | ~0.70 | ~0.95 | ~0.75 |
| Random Forest | ~0.90 | ~0.75 | ~0.82 | ~0.98 | ~0.85 |
| XGBoost | ~0.92 | ~0.80 | ~0.85 | ~0.99 | ~0.88 |
| LightGBM | ~0.91 | ~0.78 | ~0.84 | ~0.98 | ~0.87 |
| Isolation Forest | ~0.50 | ~0.50 | ~0.50 | ~0.65 | ~0.30 |

**Note**: Results vary based on dataset, preprocessing, and hyperparameters. Run training to get actual results for your data.

## 🐳 Docker Deployment

### Build and Run API

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-detection-api
```

### Using Docker Compose

```bash
# Start both API and Streamlit
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## ⚠️ Limitations and Future Work

### Current Limitations

1. **Dataset**: Uses pre-processed PCA features (original features not available)
2. **Model Selection**: Best model is selected based on ROC-AUC; may need tuning for specific business needs
3. **Real-time**: Current implementation is batch-oriented; real-time streaming would require additional infrastructure
4. **Feature Engineering**: Limited feature engineering; could benefit from domain-specific features

### Future Improvements

- [ ] Real-time streaming fraud detection
- [ ] Additional feature engineering (transaction patterns, user behavior)
- [ ] Model retraining pipeline with new data
- [ ] Integration with payment processing systems
- [ ] Advanced anomaly detection (Autoencoders, LSTM)
- [ ] A/B testing framework for model comparison
- [ ] Monitoring and alerting system
- [ ] Multi-model ensemble with voting
- [ ] Cost-sensitive learning (different costs for false positives/negatives)

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Note**: This is a demonstration project. For production use, ensure proper security measures, data privacy compliance, and thorough testing.

