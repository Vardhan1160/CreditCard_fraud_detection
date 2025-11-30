# Dataset Information

## Credit Card Fraud Detection Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

### Download Instructions

1. Visit the Kaggle dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. You will need a Kaggle account to download the dataset
3. Download the `creditcard.csv` file
4. Place it in this directory: `data/raw/creditcard.csv`

### Dataset Details

- **Size**: ~284,807 transactions
- **Features**: 30 features (28 PCA-transformed features + Time + Amount)
- **Target**: Class (0 = Normal, 1 = Fraud)
- **Class Imbalance**: Highly imbalanced (~0.172% fraud cases)

### Alternative: Synthetic Sample Data

If you don't have access to the Kaggle dataset, you can generate a synthetic sample by running:

```bash
python scripts/sample_data.py
```

This will create a small synthetic dataset at `data/raw/creditcard.csv` for demonstration purposes.

### Dataset Schema

The dataset contains the following columns:
- `Time`: Seconds elapsed between each transaction and the first transaction
- `V1-V28`: PCA-transformed features (anonymized)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = Normal, 1 = Fraud)

