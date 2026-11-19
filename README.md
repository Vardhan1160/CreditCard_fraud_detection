# Credit Card Fraud Detection 💳

A machine learning project to detect fraudulent credit card transactions using the **real Kaggle dataset**.

## About the Project

Credit card fraud causes huge financial losses every year. This project uses the **Kaggle Credit Card Fraud Detection dataset** (284,807 real transactions) to build a model that automatically identifies whether a transaction is **fraudulent** or **legitimate**.

The dataset has features **V1 to V28** (PCA-transformed for privacy), along with **Time**, **Amount**, and the target label **Class** (0 = Normal, 1 = Fraud).

Since fraud cases are very rare (**only 0.17%**), the dataset is highly imbalanced. We use:
- **SMOTE** (Synthetic Minority Oversampling) to balance the training set
- **F1 Score** and **ROC-AUC** as evaluation metrics (not plain accuracy)

## Models Used

- Logistic Regression (baseline)
- Random Forest
- XGBoost ✅ (best performance)

## Project Structure

```
CreditCard_fraud_detection/
├── data/
│   └── creditcard.csv       ← downloaded from Kaggle (284,807 rows)
├── model/
│   ├── model.pkl            ← trained best model
│   ├── scaler.pkl           ← fitted StandardScaler for Time & Amount
│   └── feature_cols.pkl     ← column order used during training
├── notebooks/
│   └── fraud_detection.ipynb ← EDA + model training notebook
├── app.py                   ← Streamlit web app
├── download_data.py         ← downloads dataset via Kaggle API
├── train.py                 ← model training script (with SMOTE)
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API (one-time)
1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. Place the downloaded `kaggle.json` at:
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

### 3. Download the dataset
```bash
python download_data.py
```
This downloads the real Kaggle dataset (~144 MB) → `data/creditcard.csv`

### 4. Train the model
```bash
python train.py
```

### 5. Launch the web app
```bash
streamlit run app.py
```
The app will open at **http://localhost:8501**

---

> **Manual download option:**  
> If you don't want to use the Kaggle API, download `creditcard.csv` manually from  
> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
> and place it in the `data/` folder, then run `python train.py`.

## Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------| 
| Logistic Regression | ~0.97    | ~0.72    | ~0.95   |
| Random Forest       | ~0.99    | ~0.85    | ~0.98   |
| XGBoost             | ~0.99    | ~0.88    | ~0.98   |

> Results on the real Kaggle dataset with SMOTE applied to the training set.

## Tech Stack

- Python 3.9+
- Scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Streamlit
- Pandas, NumPy, Plotly

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
by ULB (Université Libre de Bruxelles) Machine Learning Group.

---

Made by Vardhan Guthula | B.Tech Final Year Project
