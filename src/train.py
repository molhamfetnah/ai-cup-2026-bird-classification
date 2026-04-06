"""
Training pipeline for bird classification model.
Implements multi-label classification with ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
import os
from tqdm import tqdm


# Target columns
TARGET_COLS = ['Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese', 
               'Gulls', 'Birds of Prey', 'Waders', 'Songbirds']


def train_single_target(X_train, y_train, X_val, y_val, target_name, model_type='xgboost'):
    """Train a single model for one target column."""
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            eval_metric='logloss'
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    elif model_type == 'catboost':
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=False
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    return model


def train_ensemble_cv(feature_df, train_df, n_folds=5):
    """Train ensemble models with cross-validation."""
    
    print("Preparing data...")
    
    # Merge features with targets
    train_data = feature_df.merge(
        train_df[['track_id'] + TARGET_COLS], 
        on='track_id', 
        how='left'
    )
    
    # Separate features and targets
    X = train_data.drop(['track_id'] + TARGET_COLS, axis=1)
    feature_cols = X.columns.tolist()
    
    # Store models and predictions
    models = {target: [] for target in TARGET_COLS}
    oof_predictions = np.zeros((len(X), len(TARGET_COLS)))
    
    # Create stratified folds (using first target for stratification)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\nTraining with {n_folds}-fold cross-validation...")
    
    for target_idx, target in enumerate(TARGET_COLS):
        print(f"\n{'='*60}")
        print(f"Training models for: {target}")
        print(f"{'='*60}")
        
        y = train_data[target].values
        
        # Track fold scores
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train XGBoost
            xgb_model = train_single_target(X_train, y_train, X_val, y_val, target, 'xgboost')
            
            # Train LightGBM
            lgb_model = train_single_target(X_train, y_train, X_val, y_val, target, 'lightgbm')
            
            # Train CatBoost
            cat_model = train_single_target(X_train, y_train, X_val, y_val, target, 'catboost')
            
            # Make predictions (ensemble average)
            xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
            lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
            cat_pred = cat_model.predict_proba(X_val)[:, 1]
            
            ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
            
            # Calculate fold score
            fold_score = log_loss(y_val, ensemble_pred)
            fold_scores.append(fold_score)
            print(f"Fold {fold + 1} Log Loss: {fold_score:.6f}")
            
            # Store OOF predictions
            oof_predictions[val_idx, target_idx] = ensemble_pred
            
            # Store models
            models[target].append({
                'xgboost': xgb_model,
                'lightgbm': lgb_model,
                'catboost': cat_model
            })
        
        avg_score = np.mean(fold_scores)
        print(f"\nAverage Log Loss for {target}: {avg_score:.6f}")
    
    # Calculate overall OOF score
    overall_logloss = 0
    for target_idx, target in enumerate(TARGET_COLS):
        y_true = train_data[target].values
        y_pred = oof_predictions[:, target_idx]
        target_logloss = log_loss(y_true, y_pred)
        overall_logloss += target_logloss
        print(f"{target} OOF Log Loss: {target_logloss:.6f}")
    
    overall_logloss /= len(TARGET_COLS)
    print(f"\n{'='*60}")
    print(f"Overall OOF Log Loss: {overall_logloss:.6f}")
    print(f"{'='*60}")
    
    return models, feature_cols, oof_predictions


def predict_test(models, feature_df, feature_cols):
    """Generate predictions for test set."""
    
    print("\nGenerating test predictions...")
    
    X_test = feature_df[feature_cols]
    predictions = np.zeros((len(X_test), len(TARGET_COLS)))
    
    for target_idx, target in enumerate(TARGET_COLS):
        print(f"Predicting {target}...")
        
        target_predictions = []
        
        # Average predictions across all folds and models
        for fold_models in models[target]:
            xgb_pred = fold_models['xgboost'].predict_proba(X_test)[:, 1]
            lgb_pred = fold_models['lightgbm'].predict_proba(X_test)[:, 1]
            cat_pred = fold_models['catboost'].predict_proba(X_test)[:, 1]
            
            ensemble_pred = (xgb_pred + lgb_pred + cat_pred) / 3
            target_predictions.append(ensemble_pred)
        
        # Average across folds
        predictions[:, target_idx] = np.mean(target_predictions, axis=0)
    
    return predictions


def save_models(models, feature_cols, output_dir='models'):
    """Save trained models to disk."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving models to {output_dir}/...")
    
    # Save models
    with open(f'{output_dir}/models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
    # Save feature columns
    with open(f'{output_dir}/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("Models saved successfully!")


def load_models(model_dir='models'):
    """Load trained models from disk."""
    
    with open(f'{model_dir}/models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    with open(f'{model_dir}/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    return models, feature_cols
