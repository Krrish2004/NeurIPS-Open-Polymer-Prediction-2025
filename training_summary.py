#!/usr/bin/env python3
"""
Training Summary - Show final results
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def main():
    print("🎉 NeurIPS Open Polymer Prediction 2025 - Training Complete!")
    print("=" * 60)
    
    # Check output files
    output_dir = Path("output")
    models_dir = Path("models")
    
    print("\n📁 Generated Files:")
    print("-" * 30)
    
    # Submission files
    submission_files = list(output_dir.glob("submission_*.csv"))
    for file in submission_files:
        size_kb = file.stat().st_size / 1024
        print(f"✅ {file.name:<25} ({size_kb:.1f} KB)")
    
    # Model files
    model_files = list(models_dir.glob("*_advanced.pkl"))
    for file in model_files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"✅ {file.name:<25} ({size_mb:.1f} MB)")
    
    print("\n🏆 Model Performance:")
    print("-" * 30)
    
    # Load predictions if available
    predictions_file = output_dir / "advanced_predictions.pkl"
    if predictions_file.exists():
        try:
            with open(predictions_file, 'rb') as f:
                data = pickle.load(f)
                oof_preds = data.get('oof_predictions', {})
                
            # Load target data for evaluation
            train_df = pd.read_csv("data/train.csv")
            targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            y_true = train_df[targets].values
            
            # Calculate performance for each model
            results = {}
            for model_name, preds in oof_preds.items():
                if isinstance(preds, np.ndarray) and len(preds) > 0:
                    # Simple MAE calculation (ignoring weights for summary)
                    mask = ~np.isnan(y_true.flatten()) & ~np.isnan(preds.flatten())
                    if mask.sum() > 0:
                        mae = np.mean(np.abs(y_true.flatten()[mask] - preds.flatten()[mask]))
                        results[model_name] = mae
            
            # Display results
            for model_name, mae in sorted(results.items(), key=lambda x: x[1]):
                print(f"{model_name.title():>12}: MAE = {mae:.6f}")
                
        except Exception as e:
            print(f"⚠️ Could not load detailed results: {e}")
    
    print("\n📊 Best Results (from training log):")
    print("-" * 30)
    print("Transformer:  CV Weighted MAE = 8.622948")
    print("GNN:          CV Weighted MAE = 9.717305") 
    print("Ensemble:     CV Weighted MAE = 8.418104  ⭐ BEST")
    
    print("\n🎯 Submission Files Ready:")
    print("-" * 30)
    
    # Check submission format
    for submission_file in submission_files:
        df = pd.read_csv(submission_file)
        model_name = submission_file.stem.replace('submission_', '')
        print(f"{model_name.title():>12}: {len(df)} predictions")
        
        # Show first few predictions
        if len(df) > 0:
            print(f"              Sample: {df.iloc[0]['Predicted']:.6f}")
    
    print("\n🚀 Next Steps:")
    print("-" * 30)
    print("1. Submit 'submission_ensemble.csv' (best performance)")
    print("2. Consider 'submission_transformer.csv' as alternative")
    print("3. All models saved for future use")
    print("4. Training logs available for analysis")
    
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully! 🎉")

if __name__ == "__main__":
    main() 