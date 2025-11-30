#!/bin/bash
# Training script for credit card fraud detection

echo "Starting Credit Card Fraud Detection Training Pipeline..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run training
python scripts/train_model.py

echo "Training pipeline completed!"

