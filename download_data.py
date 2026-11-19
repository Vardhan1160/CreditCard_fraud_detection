"""
download_data.py
────────────────
Downloads the real Kaggle Credit Card Fraud Detection dataset.

Dataset : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Size    : ~144 MB  |  284,807 transactions  |  492 fraud (0.17%)

SETUP (one-time):
  1. Create a Kaggle account → https://www.kaggle.com
  2. Go to Account → API → "Create New Token"  →  downloads kaggle.json
  3. Place kaggle.json at:
       Windows : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
       Linux   : ~/.kaggle/kaggle.json
  4. pip install kaggle

Then just run:
  python download_data.py
"""

import os
import sys
import zipfile

# ── Check kaggle is installed ──────────────────────────────────────────────────
try:
    import kaggle
except ImportError:
    print("ERROR: 'kaggle' package not found.")
    print("Install it with:  pip install kaggle")
    sys.exit(1)

# ── Check kaggle.json credentials ─────────────────────────────────────────────
kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
if not os.path.exists(kaggle_json):
    print("ERROR: Kaggle API credentials not found.")
    print(f"Expected at: {kaggle_json}")
    print()
    print("Steps to fix:")
    print("  1. Go to https://www.kaggle.com → Account → API → 'Create New Token'")
    print("  2. Place the downloaded kaggle.json at:", kaggle_json)
    sys.exit(1)

# ── Download dataset ───────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)

print("=" * 55)
print("  Downloading Kaggle Credit Card Fraud Dataset")
print("=" * 55)
print("Dataset : mlg-ulb/creditcardfraud")
print("Size    : ~144 MB (please wait...)\n")

os.system("kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip")

# ── Verify download ────────────────────────────────────────────────────────────
csv_path = os.path.join('data', 'creditcard.csv')

if os.path.exists(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=5)   # quick peek
    full_df = pd.read_csv(csv_path)
    print("\n✅ Dataset downloaded successfully!")
    print(f"   File     : {csv_path}")
    print(f"   Rows     : {len(full_df):,}")
    print(f"   Columns  : {full_df.shape[1]}")
    print(f"   Fraud    : {full_df['Class'].sum():,}  ({full_df['Class'].mean()*100:.4f}%)")
    print(f"   Normal   : {(full_df['Class']==0).sum():,}")
    print("\nRun next:  python train.py")
else:
    print("\n❌ Download may have failed. Check your Kaggle credentials and internet connection.")
    print("   Or manually download from:")
    print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("   and place creditcard.csv inside the 'data/' folder.")
