#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Baseline models for polymer property prediction."""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from ..config import TARGET_COLUMNS, RANDOM_STATE


class BaselineModel:
    """Baseline model for polymer property prediction using traditional ML methods."""
    
    def __init__(self, 
                 model_type: str = 'rf',
                 target_columns: List[str] = TARGET_COLUMNS,
                 random_state: int = RANDOM_STATE):
        """
        Initialize baseline model.
        
        Args:
            model_type: Type of model to use ('rf' for RandomForest, 'gb' for GradientBoosting)
            target_columns: List of target columns to predict
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.target_columns = target_columns
        self.random_state = random_state
        self.models = {}
        self.feature_importances = {}
        
        # Initialize a model for each target
        for target in target_columns:
            if model_type == 'rf':
                self.models[target] = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=random_state,
                    n_jobs=-1
                )
            elif model_type == 'gb':
                self.models[target] = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'BaselineModel':
        """
        Train models for each target.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame
            
        Returns:
            Self
        """
        for target in self.target_columns:
            if target in y.columns:
                # Filter out rows where the target is missing
                mask = ~y[target].isna()
                
                if mask.sum() > 0:  # Only train if there are non-null values
                    print(f"Training model for {target} with {mask.sum()} samples")
                    self.models[target].fit(X[mask], y.loc[mask, target])
                    
                    # Store feature importances if available
                    if hasattr(self.models[target], 'feature_importances_'):
                        self.feature_importances[target] = pd.Series(
                            self.models[target].feature_importances_,
                            index=X.columns
                        ).sort_values(ascending=False)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all targets.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with predictions for each target
        """
        preds = pd.DataFrame(index=X.index)
        
        for target in self.target_columns:
            if target in self.models:
                preds[target] = self.models[target].predict(X)
        
        return preds
    
    def get_feature_importances(self) -> Dict[str, pd.Series]:
        """
        Get feature importances for each target.
        
        Returns:
            Dictionary of feature importance Series for each target
        """
        return self.feature_importances


class MultiTargetModel:
    """Multi-target model that uses a single model for all targets."""
    
    def __init__(self, 
                 model_type: str = 'rf',
                 target_columns: List[str] = TARGET_COLUMNS,
                 random_state: int = RANDOM_STATE):
        """
        Initialize multi-target model.
        
        Args:
            model_type: Type of model to use ('rf' for RandomForest, 'gb' for GradientBoosting)
            target_columns: List of target columns to predict
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.target_columns = target_columns
        self.random_state = random_state
        
        # Initialize base model
        if model_type == 'rf':
            base_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'gb':
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Wrap with MultiOutputRegressor
        self.model = MultiOutputRegressor(base_model)
        self.feature_importances = None
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> 'MultiTargetModel':
        """
        Train multi-target model.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame with columns for each target
            
        Returns:
            Self
        """
        # Extract target columns and handle missing values
        y_array = y[self.target_columns].values
        
        # For now, just fill missing values with mean
        for i, col in enumerate(self.target_columns):
            col_mean = np.nanmean(y_array[:, i])
            y_array[np.isnan(y_array[:, i]), i] = col_mean
        
        print(f"Training multi-target model with {len(X)} samples")
        self.model.fit(X, y_array)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for all targets.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with predictions for each target
        """
        predictions = self.model.predict(X)
        
        preds = pd.DataFrame(
            predictions,
            columns=self.target_columns,
            index=X.index
        )
        
        return preds 