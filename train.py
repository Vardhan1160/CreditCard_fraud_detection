import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Load Data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("   Credit Card Fraud Detection — Model Training (Kaggle)")
print("=" * 60)

data_path = 'data/creditcard.csv'
if not os.path.exists(data_path):
    print("\n❌ Dataset not found at data/creditcard.csv")
    print("   Run:  python download_data.py   (uses Kaggle API)")
    print("   OR manually place creditcard.csv in the data/ folder.")
    print("   Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    import sys; sys.exit(1)

df = pd.read_csv(data_path)
print(f"\nDataset loaded : {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Fraud cases    : {df['Class'].sum():,}  ({df['Class'].mean()*100:.4f}%)")
print(f"Normal cases   : {(df['Class']==0).sum():,}")
print(f"\n⚠️  Highly imbalanced dataset — using SMOTE to handle this.")

# ── Feature Engineering ───────────────────────────────────────────────────────
# Normalize Time and Amount (V1–V28 are already PCA-scaled features)
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time']   = scaler.fit_transform(df[['Time']])

# Save the scaler for use in the web app
os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

X = df.drop(columns=['Class'])
y = df['Class']

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size     : {len(X_train):,} samples")
print(f"Test size      : {len(X_test):,} samples")
print(f"Fraud in train : {y_train.sum():,}")
print(f"Fraud in test  : {y_test.sum():,}")

# ── SMOTE — Oversample fraud cases in training set only ───────────────────────
print("\nApplying SMOTE on training set...")
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"After SMOTE — Train size: {len(X_train_sm):,}  "
      f"(Fraud: {y_train_sm.sum():,}, Normal: {(y_train_sm==0).sum():,})")

# ── Train Models ──────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost'            : XGBClassifier(
                               n_estimators=200,
                               max_depth=6,
                               learning_rate=0.1,
                               scale_pos_weight=1,     # balanced after SMOTE
                               random_state=42,
                               use_label_encoder=False,
                               eval_metric='logloss',
                               verbosity=0
                           )
}

results = {}
print("\nTraining models...\n")

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_sm, y_train_sm)       # train on SMOTE-balanced data
    y_pred = model.predict(X_test)          # evaluate on REAL (imbalanced) test set
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model'   : model,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1'      : f1_score(y_test, y_pred),
        'roc_auc' : roc_auc_score(y_test, y_prob),
        'pred'    : y_pred,
    }

# ── Print Results ─────────────────────────────────────────────────────────────
print("\n" + "-" * 60)
print(f"{'Model':<25} {'Accuracy':>9} {'F1 Score':>9} {'ROC-AUC':>9}")
print("-" * 60)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>9.4f} {r['f1']:>9.4f} {r['roc_auc']:>9.4f}")
print("-" * 60)

# ── Pick Best Model (by ROC-AUC) ──────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"\n✅ Best model : {best_name}")
print(f"   ROC-AUC   : {results[best_name]['roc_auc']:.4f}")
print(f"   F1 Score  : {results[best_name]['f1']:.4f}")

# Detailed report for best model
y_pred_best = best_model.predict(X_test)
print("\nClassification Report (on real, imbalanced test set):")
print(classification_report(y_test, y_pred_best, target_names=['Normal', 'Fraud']))

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(f"  True Negatives  (Normal → Normal) : {tn:,}")
print(f"  False Positives (Normal → Fraud)  : {fp:,}")
print(f"  False Negatives (Fraud missed)    : {fn:,}")
print(f"  True Positives  (Fraud → Fraud)   : {tp:,}")

# ── Save Model & Metadata ─────────────────────────────────────────────────────
joblib.dump(best_model, 'model/model.pkl')
joblib.dump(list(X.columns), 'model/feature_cols.pkl')

print(f"\nModel saved  : model/model.pkl")
print(f"Scaler saved : model/scaler.pkl")
print(f"Training complete!")
