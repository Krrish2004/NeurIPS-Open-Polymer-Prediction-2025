#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training script for baseline models."""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import KFold

from src.config import (
    TRAIN_FILE, TEST_FILE, SAMPLE_SUBMISSION_FILE, 
    TARGET_COLUMNS, RANDOM_STATE, N_FOLDS,
    MODEL_DIR, OUTPUT_DIR
)
from src.utils.smiles_processing import extract_dataset_features, preprocess_features
from src.utils.data_processing import impute_missing_values, create_folds
from src.utils.metrics import calculate_metrics, print_metrics
from src.models.baseline import BaselineModel, MultiTargetModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train baseline models for polymer property prediction')
    
    parser.add_argument('--model_type', type=str, default='rf',
                        choices=['rf', 'gb'], 
                        help='Type of model to use (rf: Random Forest, gb: Gradient Boosting)')
    
    parser.add_argument('--separate_models', action='store_true',
                        help='Train separate models for each target')
    
    parser.add_argument('--n_folds', type=int, default=N_FOLDS,
                        help='Number of cross-validation folds')
    
    parser.add_argument('--imputation', type=str, default='none',
                        choices=['none', 'knn', 'mean', 'median'], 
                        help='Imputation method for missing values')
    
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR),
                        help='Directory to save predictions and model')
    
    return parser.parse_args()


def train_and_evaluate(args):
    """Train and evaluate models."""
    # Load data
    print(f"Loading data from {TRAIN_FILE} and {TEST_FILE}...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_FILE)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Extract features from SMILES
    print("\nExtracting molecular features from SMILES...")
    train_features = extract_dataset_features(train_df)
    test_features = extract_dataset_features(test_df)
    
    print(f"Extracted {train_features.shape[1]} features")
    
    # Preprocess features
    train_features, test_features, _ = preprocess_features(train_features, test_features, scale=True)
    
    # Create folds for cross-validation
    train_df = create_folds(train_df, n_splits=args.n_folds, random_state=RANDOM_STATE)
    
    # Initialize results DataFrame for cross-validation
    oof_predictions = pd.DataFrame(index=train_df.index)
    for col in TARGET_COLUMNS:
        oof_predictions[col] = np.nan
    
    # Cross-validation loop
    cv_metrics = []
    
    print("\nStarting cross-validation...")
    for fold in range(args.n_folds):
        print(f"\nTraining fold {fold+1}/{args.n_folds}")
        
        # Split data into train and validation
        train_idx = train_df[train_df['fold'] != fold].index
        valid_idx = train_df[train_df['fold'] == fold].index
        
        X_train = train_features.loc[train_idx]
        y_train = train_df.loc[train_idx, TARGET_COLUMNS]
        
        X_valid = train_features.loc[valid_idx]
        y_valid = train_df.loc[valid_idx, TARGET_COLUMNS]
        
        # Impute missing values in targets if specified
        if args.imputation != 'none':
            print(f"Imputing missing values with method: {args.imputation}")
            y_train = impute_missing_values(pd.DataFrame(y_train), 
                                          target_columns=TARGET_COLUMNS,
                                          method=args.imputation)
        
        # Train model
        if args.separate_models:
            print("Training separate models for each target...")
            model = BaselineModel(model_type=args.model_type, target_columns=TARGET_COLUMNS)
        else:
            print("Training multi-target model...")
            model = MultiTargetModel(model_type=args.model_type, target_columns=TARGET_COLUMNS)
        
        model.fit(X_train, y_train)
        
        # Validate
        valid_preds = model.predict(X_valid)
        oof_predictions.loc[valid_idx, TARGET_COLUMNS] = valid_preds
        
        # Calculate metrics for this fold
        fold_metrics = calculate_metrics(y_valid, valid_preds)
        print(f"\nFold {fold+1} metrics:")
        print_metrics(fold_metrics)
        
        cv_metrics.append(fold_metrics)
    
    # Calculate average metrics across folds
    avg_wmae = np.mean([m['weighted_mae'] for m in cv_metrics])
    print(f"\nAverage weighted MAE across {args.n_folds} folds: {avg_wmae:.4f}")
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    if args.separate_models:
        final_model = BaselineModel(model_type=args.model_type, target_columns=TARGET_COLUMNS)
    else:
        final_model = MultiTargetModel(model_type=args.model_type, target_columns=TARGET_COLUMNS)
    
    # Impute missing values in targets if specified
    if args.imputation != 'none':
        print(f"Imputing missing values with method: {args.imputation}")
        train_targets = impute_missing_values(train_df[TARGET_COLUMNS], 
                                            target_columns=TARGET_COLUMNS,
                                            method=args.imputation)
    else:
        train_targets = train_df[TARGET_COLUMNS]
    
    final_model.fit(train_features, train_targets)
    
    # Make predictions on test data
    print("\nGenerating test predictions...")
    test_preds = final_model.predict(test_features)
    
    # Create submission file
    submission_df[TARGET_COLUMNS] = test_preds
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions
    oof_file = Path(args.output_dir) / f"oof_preds_{args.model_type}.csv"
    submission_file = Path(args.output_dir) / f"submission_{args.model_type}.csv"
    
    oof_predictions.to_csv(oof_file)
    submission_df.to_csv(submission_file, index=False)
    
    print(f"Saved out-of-fold predictions to {oof_file}")
    print(f"Saved test predictions to {submission_file}")
    
    # Save model
    model_type = args.model_type
    model_approach = "separate" if args.separate_models else "multi"
    model_file = Path(MODEL_DIR) / f"baseline_{model_type}_{model_approach}.pkl"
    
    joblib.dump(final_model, model_file)
    print(f"Saved model to {model_file}")
    
    return final_model, oof_predictions, submission_df


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args) 