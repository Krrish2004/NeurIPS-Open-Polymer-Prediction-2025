#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics utilities for polymer property prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.metrics import mean_absolute_error, r2_score

# Import from parent config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMNS


def calculate_metrics(y_true: pd.DataFrame, 
                      y_pred: pd.DataFrame, 
                      target_columns: List[str] = TARGET_COLUMNS) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Calculate evaluation metrics for each target.
    
    Args:
        y_true: DataFrame with true values
        y_pred: DataFrame with predicted values
        target_columns: List of target columns
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # For each target, calculate metrics
    for col in target_columns:
        if col in y_true.columns and col in y_pred.columns:
            # Filter out missing values
            mask = ~np.isnan(y_true[col])
            
            if mask.sum() > 0:  # Only calculate if there are non-null values
                true_values = y_true.loc[mask, col]
                pred_values = y_pred.loc[mask, col]
                
                # Calculate metrics
                mae = mean_absolute_error(true_values, pred_values)
                r2 = r2_score(true_values, pred_values)
                
                metrics[col] = {
                    'mae': mae,
                    'r2': r2
                }
                
    # Calculate weighted MAE
    # Extract arrays for all targets
    true_array = np.zeros((len(y_true), len(target_columns)))
    pred_array = np.zeros((len(y_pred), len(target_columns)))
    
    for i, col in enumerate(target_columns):
        if col in y_true.columns:
            true_array[:, i] = y_true[col].values
        if col in y_pred.columns:
            pred_array[:, i] = y_pred[col].values
    
    metrics['weighted_mae'] = weighted_mae(true_array, pred_array)
    
    return metrics


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
    # Create masks for each target column
    masks = ~np.isnan(y_true)
    
    # Calculate errors for non-missing values
    errors = np.zeros_like(y_true)
    errors[masks] = np.abs(y_true[masks] - y_pred[masks])
    
    # Calculate standard deviation for each target
    stds = np.array([np.nanstd(y_true[:, i]) for i in range(y_true.shape[1])])
    weights = 1.0 / (stds + 1e-8)  # Add small epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    # Compute weighted MAE for each sample
    weighted_errors = np.zeros_like(y_true)
    for i in range(y_true.shape[1]):
        weighted_errors[:, i] = errors[:, i] * weights[i]
    
    # Average across all non-missing values
    wmae = np.nanmean(weighted_errors)
    
    return wmae


def print_metrics(metrics: Dict[str, Union[float, Dict[str, float]]]) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\nMetrics Summary:")
    print("=" * 40)
    
    # Print per-target metrics
    for target, target_metrics in metrics.items():
        if isinstance(target_metrics, dict):
            print(f"\nTarget: {target}")
            print("-" * 30)
            for metric_name, value in target_metrics.items():
                print(f"{metric_name}: {value:.4f}")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    print("-" * 30)
    
    if 'weighted_mae' in metrics:
        print(f"Weighted MAE: {metrics['weighted_mae']:.4f}")
    
    print("=" * 40)


def calculate_weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, property_weights: dict = None) -> float:
    """
    Calculate weighted MAE with specified property weights.
    
    Args:
        y_true: True values (n_samples, n_targets)
        y_pred: Predicted values (n_samples, n_targets)
        property_weights: Dictionary of property weights
        
    Returns:
        Weighted MAE score
    """
    if property_weights is None:
        # Use default weights
        property_weights = {
            'Tg': 0.15,
            'FFV': 0.2, 
            'Tc': 0.2,
            'Density': 0.2,
            'Rg': 0.25
        }
    
    # Convert to arrays if needed
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Handle 1D arrays by reshaping
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Calculate MAE for each target
    maes = []
    weights_list = []
    target_names = list(property_weights.keys())
    
    for i, target in enumerate(target_names):
        if i < y_true.shape[1]:
            # Get valid (non-NaN) values
            mask = ~np.isnan(y_true[:, i])
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_true[mask, i] - y_pred[mask, i]))
                maes.append(mae)
                weights_list.append(property_weights[target])
    
    if not maes:
        return 0.0
    
    # Calculate weighted average
    maes = np.array(maes)
    weights_list = np.array(weights_list)
    weighted_mae = np.average(maes, weights=weights_list)
    
    return weighted_mae 