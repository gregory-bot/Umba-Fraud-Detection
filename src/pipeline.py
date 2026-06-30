import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import load_data, preprocess
from model_training import run_model_comparison
import pandas as pd
import numpy as np


def main():
    print("Loading data...")
    train, test, identity = load_data('data')

    print(f"Train: {train.shape}, Test: {test.shape}, Identity: {identity.shape}")

    print("\nPreprocessing and feature engineering...")
    X_train, X_test, y = preprocess(train, test, identity)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Fraud rate: {y.mean():.4f}")

    print("\nRunning model comparison with time-series cross-validation...")
    final_model, test_probs, results, best_model = run_model_comparison(X_train, X_test, y)

    submission = pd.DataFrame({
        'TransactionID': test['TransactionID'].values,
        'isFraud_prob': test_probs
    })
    submission.to_csv('predictions.csv', index=False)
    print(f"\nPredictions saved to predictions.csv")
    print(f"Shape: {submission.shape}")
    print(f"Predicted fraud rate: {test_probs.mean():.4f}")
    print(f"Prediction range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")

    return final_model, test_probs, results, best_model


if __name__ == '__main__':
    main()
