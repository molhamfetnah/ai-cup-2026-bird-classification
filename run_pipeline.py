"""
Main pipeline to train models and generate submission.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from features import extract_all_features
from train import train_ensemble_cv, predict_test, save_models, TARGET_COLS


def main():
    print("="*60)
    print("Bird Classification Pipeline")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Extract features
    print("\n2. Extracting features from training data...")
    train_features = extract_all_features(train_df)
    train_features.to_csv('outputs/train_features.csv', index=False)
    print(f"Train features saved: {train_features.shape}")
    
    print("\n3. Extracting features from test data...")
    test_features = extract_all_features(test_df)
    test_features.to_csv('outputs/test_features.csv', index=False)
    print(f"Test features saved: {test_features.shape}")
    
    # Train models
    print("\n4. Training ensemble models...")
    models, feature_cols, oof_preds = train_ensemble_cv(train_features, train_df, n_folds=5)
    
    # Save models
    save_models(models, feature_cols)
    
    # Save OOF predictions
    oof_df = train_df[['track_id']].copy()
    for idx, col in enumerate(TARGET_COLS):
        oof_df[col] = oof_preds[:, idx]
    oof_df.to_csv('outputs/oof_predictions.csv', index=False)
    print("\nOOF predictions saved to outputs/oof_predictions.csv")
    
    # Generate test predictions
    print("\n5. Generating test predictions...")
    test_preds = predict_test(models, test_features, feature_cols)
    
    # Create submission
    submission = pd.DataFrame({
        'track_id': test_df['track_id']
    })
    
    for idx, col in enumerate(TARGET_COLS):
        submission[col] = test_preds[:, idx]
    
    # Save submission
    submission.to_csv('outputs/submission.csv', index=False)
    print("\n" + "="*60)
    print("Submission saved to outputs/submission.csv")
    print("="*60)
    
    # Display sample predictions
    print("\nSample predictions:")
    print(submission.head(10))
    
    return submission


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run pipeline
    submission = main()
