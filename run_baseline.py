"""
Quick baseline model for initial submission.
Uses simple features without trajectory parsing for faster execution.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import xgboost as xgb
import os


TARGET_COLS = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 
               'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']


def create_target_columns(df):
    """Create binary target columns from bird_group."""
    for target in TARGET_COLS:
        df[target] = (df['bird_group'] == target).astype(int)
    return df


def extract_simple_features(df):
    """Extract simple features without trajectory parsing."""
    
    features = pd.DataFrame()
    features['track_id'] = df['track_id']
    
    # Radar features - convert to numeric
    features['radar_bird_size'] = pd.to_numeric(df['radar_bird_size'], errors='coerce').fillna(0)
    features['airspeed'] = pd.to_numeric(df['airspeed'], errors='coerce').fillna(0)
    features['min_z'] = pd.to_numeric(df['min_z'], errors='coerce').fillna(0)
    features['max_z'] = pd.to_numeric(df['max_z'], errors='coerce').fillna(0)
    features['altitude_range'] = features['max_z'] - features['min_z']
    
    # Temporal features
    df['timestamp_start'] = pd.to_datetime(df['timestamp_start_radar_utc'], errors='coerce')
    df['timestamp_end'] = pd.to_datetime(df['timestamp_end_radar_utc'], errors='coerce')
    
    features['duration_seconds'] = (df['timestamp_end'] - df['timestamp_start']).dt.total_seconds().fillna(0)
    features['hour_of_day'] = df['timestamp_start'].dt.hour.fillna(0)
    features['day_of_week'] = df['timestamp_start'].dt.dayofweek.fillna(0)
    features['month'] = df['timestamp_start'].dt.month.fillna(0)
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
    
    # Trajectory length (approximate from hex string length)
    features['trajectory_length'] = df['trajectory'].fillna('').str.len()
    
    # Observer info (only available in train)
    if 'n_birds_observed' in df.columns:
        features['n_birds_observed'] = pd.to_numeric(df['n_birds_observed'], errors='coerce').fillna(0)
    else:
        features['n_birds_observed'] = 0
    
    # Fill any remaining NaN values
    features = features.fillna(0)
    
    return features


def train_baseline(train_df, n_folds=3):
    """Train baseline XGBoost models."""
    
    print("Creating target columns...")
    train_df = create_target_columns(train_df)
    
    print("Extracting simple features...")
    features = extract_simple_features(train_df)
    
    X = features.drop('track_id', axis=1)
    feature_cols = X.columns.tolist()
    
    models = []
    oof_preds = np.zeros((len(X), len(TARGET_COLS)))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for target_idx, target in enumerate(TARGET_COLS):
        print(f"\nTraining {target}...")
        
        y = train_df[target].values
        target_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                tree_method='hist',
                early_stopping_rounds=20
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            pred = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx, target_idx] = pred
            
            target_models.append(model)
            
        models.append(target_models)
        
        # Calculate OOF score
        logloss = log_loss(y, oof_preds[:, target_idx])
        print(f"{target} OOF Log Loss: {logloss:.6f}")
    
    avg_logloss = np.mean([log_loss(train_df[t].values, oof_preds[:, i]) 
                           for i, t in enumerate(TARGET_COLS)])
    print(f"\nAverage OOF Log Loss: {avg_logloss:.6f}")
    
    return models, feature_cols


def predict_baseline(models, test_df, feature_cols):
    """Generate predictions with baseline models."""
    
    features = extract_simple_features(test_df)
    X_test = features[feature_cols]
    
    predictions = np.zeros((len(X_test), len(TARGET_COLS)))
    
    for target_idx, target_models in enumerate(models):
        preds = []
        for model in target_models:
            pred = model.predict_proba(X_test)[:, 1]
            preds.append(pred)
        predictions[:, target_idx] = np.mean(preds, axis=0)
    
    return predictions


def main():
    print("Baseline Model Pipeline")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Train
    models, feature_cols = train_baseline(train_df, n_folds=3)
    
    # Predict
    print("\nGenerating test predictions...")
    test_preds = predict_baseline(models, test_df, feature_cols)
    
    # Create submission
    submission = pd.DataFrame({'track_id': test_df['track_id']})
    for idx, col in enumerate(TARGET_COLS):
        submission[col] = test_preds[:, idx]
    
    # Save
    os.makedirs('outputs', exist_ok=True)
    submission.to_csv('outputs/baseline_submission.csv', index=False)
    
    print("\n" + "="*60)
    print("Baseline submission saved!")
    print("="*60)
    print("\nSample:")
    print(submission.head())


if __name__ == '__main__':
    main()
