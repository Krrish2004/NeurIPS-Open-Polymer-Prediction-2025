#!/usr/bin/env python3
"""
Advanced Model Training Pipeline for NeurIPS Open Polymer Prediction 2025

This script implements:
1. Transformer-based molecular property prediction
2. Graph Neural Network (GNN) models
3. Model ensemble methods
4. Hyperparameter optimization
5. Advanced evaluation and submission generation
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path

# Add src to path
sys.path.append('src')

from config import Config
from utils.data_processing import DataProcessor
from utils.metrics import calculate_weighted_mae, calculate_metrics
from models.baseline import BaselineModel
from models.transformer import TransformerRegressor
from models.gnn import GNNRegressor

import optuna
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


class AdvancedModelPipeline:
    """Advanced model training pipeline with ensemble and optimization."""
    
    def __init__(self, config_path=None):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.models = {}
        self.ensemble_model = None
        self.oof_predictions = {}
        self.test_predictions = {}
        
    def load_data(self):
        """Load and preprocess data."""
        print("Loading data...")
        
        # Load raw data
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Process data
        self.X_train, self.y_train = self.data_processor.prepare_data(train_df, is_training=True)
        self.X_test, _ = self.data_processor.prepare_data(test_df, is_training=False)
        
        print(f"Features extracted: {self.X_train.shape[1]} features")
        print(f"Training samples: {len(self.X_train)}")
        
        return self.X_train, self.y_train, self.X_test
    
    def train_transformer_model(self, X, y, hyperparams=None):
        """Train transformer model with optional hyperparameters."""
        print("\n=== Training Transformer Model ===")
        
        if hyperparams is None:
            hyperparams = {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'batch_size': 32,
                'epochs': 50  # Reduced for faster training
            }
        
        # Get SMILES strings for transformer
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        
        smiles_train = train_df['SMILES'].values
        smiles_test = test_df['SMILES'].values
        
        model = TransformerRegressor(**hyperparams)
        
        # Cross-validation
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        oof_preds = np.full_like(y, np.nan)
        test_preds = []
        
        start_time = time.time()
        fold_times = []
        
        with tqdm(total=self.config.cv_folds, desc="Transformer CV", unit="fold") as pbar:
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                fold_start = time.time()
                
                X_fold_train = smiles_train[train_idx]
                y_fold_train = y[train_idx]
                X_fold_val = smiles_train[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Predict validation set
                val_preds = model.predict(X_fold_val)
                oof_preds[val_idx] = val_preds
                
                # Predict test set
                test_pred = model.predict(smiles_test)
                test_preds.append(test_pred)
                
                # Calculate metrics for this fold
                fold_weighted_mae = calculate_weighted_mae(y[val_idx], val_preds, self.config.property_weights)
                
                fold_time = time.time() - fold_start
                fold_times.append(fold_time)
                
                # Estimate remaining time
                avg_fold_time = np.mean(fold_times)
                remaining_folds = self.config.cv_folds - (fold + 1)
                eta_seconds = avg_fold_time * remaining_folds
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
                
                pbar.set_postfix({
                    'MAE': f'{fold_weighted_mae:.6f}',
                    'Time': f'{fold_time:.1f}s',
                    'ETA': eta_str
                })
                pbar.update(1)
        
        # Average test predictions
        test_pred_avg = np.mean(test_preds, axis=0)
        
        # Calculate overall metrics
        overall_weighted_mae = calculate_weighted_mae(y, oof_preds, self.config.property_weights)
        print(f"\nTransformer CV Weighted MAE: {overall_weighted_mae:.6f}")
        
        self.models['transformer'] = model
        self.oof_predictions['transformer'] = oof_preds
        self.test_predictions['transformer'] = test_pred_avg
        
        return overall_weighted_mae
    
    def train_gnn_model(self, X, y, hyperparams=None):
        """Train GNN model with optional hyperparameters."""
        print("\n=== Training GNN Model ===")
        
        if hyperparams is None:
            hyperparams = {
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'pool': 'mean',
                'learning_rate': 1e-3,
                'batch_size': 32,
                'epochs': 50  # Reduced for faster training
            }
        
        # Get SMILES strings for GNN
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)
        
        smiles_train = train_df['SMILES'].values
        smiles_test = test_df['SMILES'].values
        
        model = GNNRegressor(**hyperparams)
        
        # Cross-validation
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        oof_preds = np.full_like(y, np.nan)
        test_preds = []
        
        start_time = time.time()
        fold_times = []
        
        with tqdm(total=self.config.cv_folds, desc="GNN CV", unit="fold") as pbar:
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                fold_start = time.time()
                
                X_fold_train = smiles_train[train_idx]
                y_fold_train = y[train_idx]
                X_fold_val = smiles_train[val_idx]
                
                # Train model
                model.fit(X_fold_train, y_fold_train)
                
                # Predict validation set
                val_preds = model.predict(X_fold_val)
                oof_preds[val_idx] = val_preds
                
                # Predict test set
                test_pred = model.predict(smiles_test)
                test_preds.append(test_pred)
                
                # Calculate metrics for this fold
                fold_weighted_mae = calculate_weighted_mae(y[val_idx], val_preds, self.config.property_weights)
                
                fold_time = time.time() - fold_start
                fold_times.append(fold_time)
                
                # Estimate remaining time
                avg_fold_time = np.mean(fold_times)
                remaining_folds = self.config.cv_folds - (fold + 1)
                eta_seconds = avg_fold_time * remaining_folds
                eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s" if eta_seconds > 60 else f"{int(eta_seconds)}s"
                
                pbar.set_postfix({
                    'MAE': f'{fold_weighted_mae:.6f}',
                    'Time': f'{fold_time:.1f}s',
                    'ETA': eta_str
                })
                pbar.update(1)
        
        # Average test predictions
        test_pred_avg = np.mean(test_preds, axis=0)
        
        # Calculate overall metrics
        overall_weighted_mae = calculate_weighted_mae(y, oof_preds, self.config.property_weights)
        print(f"\nGNN CV Weighted MAE: {overall_weighted_mae:.6f}")
        
        self.models['gnn'] = model
        self.oof_predictions['gnn'] = oof_preds
        self.test_predictions['gnn'] = test_pred_avg
        
        return overall_weighted_mae
    
    def create_ensemble(self, X, y):
        """Create ensemble model from trained individual models."""
        print("\n=== Creating Ensemble Model ===")
        
        if not self.oof_predictions:
            print("No models trained yet. Train individual models first.")
            return
        
        # Stack out-of-fold predictions
        oof_features = []
        test_features = []
        
        for model_name, oof_pred in self.oof_predictions.items():
            oof_features.append(oof_pred)
            test_features.append(self.test_predictions[model_name])
        
        oof_features = np.hstack(oof_features)  # Shape: (n_samples, n_models * n_targets)
        test_features = np.hstack(test_features)
        
        # Train meta-model (simple averaging for now)
        ensemble_oof = np.mean([pred for pred in self.oof_predictions.values()], axis=0)
        ensemble_test = np.mean([pred for pred in self.test_predictions.values()], axis=0)
        
        # Calculate ensemble metrics
        ensemble_weighted_mae = calculate_weighted_mae(y, ensemble_oof, self.config.property_weights)
        print(f"Ensemble CV Weighted MAE: {ensemble_weighted_mae:.6f}")
        
        self.oof_predictions['ensemble'] = ensemble_oof
        self.test_predictions['ensemble'] = ensemble_test
        
        return ensemble_weighted_mae
    
    def optimize_hyperparameters(self, model_type='transformer', n_trials=20):
        """Optimize hyperparameters using Optuna."""
        print(f"\n=== Optimizing {model_type.title()} Hyperparameters ===")
        
        def objective(trial):
            if model_type == 'transformer':
                hyperparams = {
                    'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
                    'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
                    'num_layers': trial.suggest_int('num_layers', 3, 8),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'epochs': 20  # Reduced for optimization
                }
                
                score = self.train_transformer_model(self.X_train, self.y_train, hyperparams)
                
            elif model_type == 'gnn':
                hyperparams = {
                    'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                    'num_layers': trial.suggest_int('num_layers', 2, 5),
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                    'pool': trial.suggest_categorical('pool', ['mean', 'add']),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                    'epochs': 20  # Reduced for optimization
                }
                
                score = self.train_gnn_model(self.X_train, self.y_train, hyperparams)
            
            return score
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"Best {model_type} hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best CV Score: {study.best_value:.6f}")
        
        return study.best_params
    
    def generate_submission(self, submission_name='ensemble'):
        """Generate submission file in correct competition format."""
        print(f"\n=== Generating Submission ({submission_name}) ===")
        
        if submission_name not in self.test_predictions:
            print(f"No predictions found for {submission_name}")
            return
        
        test_pred = self.test_predictions[submission_name]
        
        # Convert to numpy array if needed
        if not isinstance(test_pred, np.ndarray):
            test_pred = np.array(test_pred)
        
        # Load test data to get IDs
        test_df = pd.read_csv(self.config.test_path)
        test_ids = test_df['id'].values
        
        # Ensure predictions are in correct shape (n_samples, n_targets)
        if test_pred.ndim == 1:
            if len(test_pred) == len(test_ids) * 5:  # 5 targets per sample
                test_pred = test_pred.reshape(len(test_ids), 5)
            else:
                print(f"⚠️ Unexpected prediction length: {len(test_pred)}")
                return
        elif test_pred.ndim == 2:
            if test_pred.shape[0] != len(test_ids) or test_pred.shape[1] != 5:
                print(f"⚠️ Unexpected prediction shape: {test_pred.shape}")
                return
        else:
            print(f"⚠️ Unexpected prediction dimensions: {test_pred.ndim}")
            return
        
        # Target columns in correct order
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Create submission DataFrame in competition format
        submission_data = {'id': test_ids}
        for i, col in enumerate(target_columns):
            submission_data[col] = test_pred[:, i]
        
        submission_df = pd.DataFrame(submission_data)
        
        # Save submission
        output_file = self.config.output_dir / f'submission_{submission_name}.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        # Show sample predictions
        print("Sample predictions:")
        print(submission_df.head())
        
        return submission_df
    
    def save_models(self):
        """Save all trained models."""
        print("\n=== Saving Models ===")
        
        for model_name, model in self.models.items():
            model_path = self.config.models_dir / f'{model_name}_advanced.pkl'
            
            if hasattr(model, 'save'):
                model.save(str(model_path))
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"Saved {model_name} model to {model_path}")
        
        # Save predictions
        predictions_path = self.config.output_dir / 'advanced_predictions.pkl'
        with open(predictions_path, 'wb') as f:
            pickle.dump({
                'oof_predictions': self.oof_predictions,
                'test_predictions': self.test_predictions
            }, f)
        
        print(f"Saved predictions to {predictions_path}")
    
    def create_performance_summary(self):
        """Create performance summary and visualization."""
        print("\n=== Performance Summary ===")
        
        results = {}
        for model_name, oof_pred in self.oof_predictions.items():
            weighted_mae = calculate_weighted_mae(self.y_train, oof_pred, self.config.property_weights)
            results[model_name] = weighted_mae
            print(f"{model_name.title():>12}: {weighted_mae:.6f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Performance comparison
        plt.subplot(1, 2, 1)
        models = list(results.keys())
        scores = list(results.values())
        
        bars = plt.bar(models, scores)
        plt.title('Model Performance Comparison')
        plt.ylabel('Weighted MAE (CV)')
        plt.xticks(rotation=45)
        
        # Color bars
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)
        
        # Prediction correlation
        plt.subplot(1, 2, 2)
        if len(self.oof_predictions) >= 2:
            model_names = list(self.oof_predictions.keys())
            if len(model_names) >= 2:
                pred1 = self.oof_predictions[model_names[0]].flatten()
                pred2 = self.oof_predictions[model_names[1]].flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(pred1) | np.isnan(pred2))
                pred1_clean = pred1[mask]
                pred2_clean = pred2[mask]
                
                plt.scatter(pred1_clean, pred2_clean, alpha=0.5)
                plt.xlabel(f'{model_names[0].title()} Predictions')
                plt.ylabel(f'{model_names[1].title()} Predictions')
                plt.title('Model Prediction Correlation')
                
                # Add correlation coefficient
                corr = np.corrcoef(pred1_clean, pred2_clean)[0, 1]
                plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                        transform=plt.gca().transAxes, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.config.output_dir / 'advanced_performance_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results


def main():
    """Main training pipeline."""
    print("NeurIPS Open Polymer Prediction 2025 - Advanced Model Training")
    print("=" * 70)
    
    overall_start = time.time()
    
    # Initialize pipeline
    pipeline = AdvancedModelPipeline()
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_test = pipeline.load_data()
    
    # Training options
    train_transformer = True
    train_gnn = True
    optimize_hyperparams = False  # Set to True for hyperparameter optimization
    create_ensemble = True
    
    # Estimate total steps for progress
    total_steps = 0
    if train_transformer: total_steps += 1
    if train_gnn: total_steps += 1
    if create_ensemble: total_steps += 1
    total_steps += 2  # save models + performance summary
    
    try:
        with tqdm(total=total_steps, desc="Overall Progress", unit="step", position=0) as main_pbar:
            
            # Train individual models
            if train_transformer:
                try:
                    main_pbar.set_description("Training Transformer")
                    pipeline.train_transformer_model(X_train, y_train)
                    main_pbar.update(1)
                except Exception as e:
                    print(f"Transformer training failed: {e}")
                    print("Continuing with other models...")
                    main_pbar.update(1)
            
            if train_gnn:
                try:
                    main_pbar.set_description("Training GNN")
                    pipeline.train_gnn_model(X_train, y_train)
                    main_pbar.update(1)
                except Exception as e:
                    print(f"GNN training failed: {e}")
                    print("Continuing with other models...")
                    main_pbar.update(1)
            
            # Hyperparameter optimization (optional)
            if optimize_hyperparams and pipeline.models:
                for model_type in ['transformer', 'gnn']:
                    if model_type in [name.split('_')[0] for name in pipeline.models.keys()]:
                        try:
                            main_pbar.set_description(f"Optimizing {model_type}")
                            best_params = pipeline.optimize_hyperparameters(model_type, n_trials=10)
                            print(f"Best {model_type} params: {best_params}")
                        except Exception as e:
                            print(f"Hyperparameter optimization for {model_type} failed: {e}")
            
            # Create ensemble
            if create_ensemble and len(pipeline.oof_predictions) > 1:
                main_pbar.set_description("Creating Ensemble")
                pipeline.create_ensemble(X_train, y_train)
                main_pbar.update(1)
            
            # Generate submissions
            main_pbar.set_description("Generating Submissions")
            for model_name in pipeline.test_predictions.keys():
                pipeline.generate_submission(model_name)
            
            # Save models and results
            main_pbar.set_description("Saving Models")
            pipeline.save_models()
            main_pbar.update(1)
            
            # Create performance summary
            main_pbar.set_description("Creating Summary")
            pipeline.create_performance_summary()
            main_pbar.update(1)
        
        total_time = time.time() - overall_start
        
        print("\n" + "=" * 70)
        print("🎉 Advanced model training completed successfully!")
        print(f"⏱️  Total training time: {int(total_time//60)}m {int(total_time%60)}s")
        print(f"📁 Models saved in: {pipeline.config.models_dir}")
        print(f"📁 Outputs saved in: {pipeline.config.output_dir}")
        print("=" * 70)
        
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 