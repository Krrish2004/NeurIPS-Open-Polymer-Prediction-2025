#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data processing utilities for polymer property prediction."""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from ..config import TARGET_COLUMNS, RANDOM_STATE


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test datasets.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def impute_missing_values(df: pd.DataFrame, 
                          target_columns: List[str] = TARGET_COLUMNS,
                          method: str = 'knn', 
                          n_neighbors: int = 5) -> pd.DataFrame:
    """
    Impute missing values in the dataset.
    
    Args:
        df: DataFrame with missing values
        target_columns: List of target columns to impute
        method: Imputation method ('knn', 'mean', 'median')
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        DataFrame with imputed values
    """
    df_copy = df.copy()
    
    # Extract only the target columns for imputation
    targets = df_copy[target_columns].copy()
    
    if method == 'knn':
        # KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_targets = imputer.fit_transform(targets)
        df_copy[target_columns] = imputed_targets
    elif method == 'mean':
        # Mean imputation
        df_copy[target_columns] = df_copy[target_columns].fillna(df_copy[target_columns].mean())
    elif method == 'median':
        # Median imputation
        df_copy[target_columns] = df_copy[target_columns].fillna(df_copy[target_columns].median())
    else:
        raise ValueError(f"Unsupported imputation method: {method}")
    
    return df_copy


def create_folds(df: pd.DataFrame, 
                 n_splits: int = 5, 
                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Create cross-validation folds.
    
    Args:
        df: DataFrame
        n_splits: Number of folds
        random_state: Random seed
        
    Returns:
        DataFrame with 'fold' column
    """
    df_copy = df.copy()
    df_copy['fold'] = -1
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Assign folds
    for i, (_, val_idx) in enumerate(kf.split(df_copy)):
        df_copy.loc[val_idx, 'fold'] = i
    
    return df_copy


def normalize_targets(df: pd.DataFrame, 
                      target_columns: List[str] = TARGET_COLUMNS) -> Tuple[pd.DataFrame, Dict[str, StandardScaler]]:
    """
    Normalize target variables.
    
    Args:
        df: DataFrame
        target_columns: List of target columns
        
    Returns:
        DataFrame with normalized targets and dictionary of scalers for each target
    """
    df_copy = df.copy()
    scalers = {}
    
    for col in target_columns:
        if col in df_copy.columns:
            # Apply only to non-null values
            non_null_mask = df_copy[col].notnull()
            
            if non_null_mask.sum() > 0:  # Only normalize if there are non-null values
                scaler = StandardScaler()
                df_copy.loc[non_null_mask, col] = scaler.fit_transform(
                    df_copy.loc[non_null_mask, col].values.reshape(-1, 1)
                ).flatten()
                
                scalers[col] = scaler
    
    return df_copy, scalers


def denormalize_targets(df: pd.DataFrame, 
                        scalers: Dict[str, StandardScaler], 
                        target_columns: List[str] = TARGET_COLUMNS) -> pd.DataFrame:
    """
    Denormalize target variables.
    
    Args:
        df: DataFrame with normalized targets
        scalers: Dictionary of scalers for each target
        target_columns: List of target columns
        
    Returns:
        DataFrame with denormalized targets
    """
    df_copy = df.copy()
    
    for col in target_columns:
        if col in df_copy.columns and col in scalers:
            df_copy[col] = scalers[col].inverse_transform(
                df_copy[col].values.reshape(-1, 1)
            ).flatten()
    
    return df_copy


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate weighted Mean Absolute Error.
    
    The weights are the inverse of the standard deviation of each target.
    
    Args:
        y_true: True values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        
    Returns:
        Weighted MAE score
    """
    # Handle missing values by ignoring them
    mask = ~np.isnan(y_true)
    
    errors = np.abs(y_true[mask] - y_pred[mask])
    
    # Calculate standard deviation for each target
    stds = np.nanstd(y_true, axis=0)
    weights = 1.0 / (stds + 1e-8)  # Add small epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted MAE
    weighted_errors = errors * weights[np.newaxis, :]
    wmae = np.nanmean(weighted_errors)
    
    return wmae 