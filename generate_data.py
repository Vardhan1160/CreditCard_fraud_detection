import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

print("Generating synthetic credit card transaction data...")

n_samples = 10000
fraud_ratio = 0.02   # 2% fraud cases (a bit higher so model can learn)

n_fraud = int(n_samples * fraud_ratio)
n_normal = n_samples - n_fraud

# --- Generate normal transactions ---
normal_data = {}
for i in range(1, 29):
    normal_data[f'V{i}'] = np.random.normal(0, 1, n_normal)

normal_data['Time']   = np.random.uniform(0, 172800, n_normal)   # 0 to 48 hours in seconds
normal_data['Amount'] = np.abs(np.random.lognormal(mean=3.0, sigma=1.5, size=n_normal))
normal_data['Class']  = 0

# --- Generate fraud transactions ---
fraud_data = {}
for i in range(1, 29):
    if i in [1, 2, 3, 4, 5]:
        # Make a few features clearly different for fraud
        fraud_data[f'V{i}'] = np.random.normal(-3, 1.5, n_fraud)
    else:
        fraud_data[f'V{i}'] = np.random.normal(0, 1.5, n_fraud)

fraud_data['Time']   = np.random.uniform(0, 172800, n_fraud)
fraud_data['Amount'] = np.abs(np.random.lognormal(mean=4.5, sigma=2.0, size=n_fraud))
fraud_data['Class']  = 1

# --- Combine and shuffle ---
df_normal = pd.DataFrame(normal_data)
df_fraud  = pd.DataFrame(fraud_data)
df = pd.concat([df_normal, df_fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Reorder columns to match the standard creditcard.csv format
cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
df = df[cols]

# Save to data folder
os.makedirs('data', exist_ok=True)
df.to_csv('data/creditcard.csv', index=False)

print(f"Done!")
print(f"Total transactions : {len(df)}")
print(f"Normal transactions: {(df['Class'] == 0).sum()}")
print(f"Fraud transactions : {(df['Class'] == 1).sum()}")
print(f"Fraud percentage   : {df['Class'].mean() * 100:.2f}%")
print(f"Saved to: data/creditcard.csv")
