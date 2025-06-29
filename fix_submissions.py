#!/usr/bin/env python3
"""
Fix submission format to match sample_submission.csv structure
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def fix_submission_format():
    """Convert predictions to correct submission format."""
    
    # Load test data to get IDs
    test_df = pd.read_csv('test.csv')
    test_ids = test_df['id'].values
    
    print(f"Test IDs: {test_ids}")
    print(f"Number of test samples: {len(test_ids)}")
    
    # Load advanced predictions
    predictions_file = Path('output/advanced_predictions.pkl')
    if not predictions_file.exists():
        print("❌ Advanced predictions not found!")
        return
    
    with open(predictions_file, 'rb') as f:
        data = pickle.load(f)
        test_predictions = data.get('test_predictions', {})
    
    # Target columns in correct order
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Process each model's predictions
    for model_name, predictions in test_predictions.items():
        print(f"\n=== Processing {model_name} ===")
        
        # Convert to numpy array
        pred_array = np.array(predictions)
        print(f"Prediction shape: {pred_array.shape}")
        
        # Handle different prediction formats
        if pred_array.ndim == 1:
            # Flatten predictions - reshape to (n_samples, n_targets)
            if len(pred_array) == 15:  # 3 samples × 5 targets
                pred_array = pred_array.reshape(3, 5)
            else:
                print(f"⚠️ Unexpected prediction length: {len(pred_array)}")
                continue
        elif pred_array.ndim == 2:
            # Already in correct shape
            if pred_array.shape[0] != 3 or pred_array.shape[1] != 5:
                print(f"⚠️ Unexpected prediction shape: {pred_array.shape}")
                continue
        else:
            print(f"⚠️ Unexpected prediction dimensions: {pred_array.ndim}")
            continue
        
        # Create submission DataFrame
        submission_data = {'id': test_ids}
        for i, col in enumerate(target_columns):
            submission_data[col] = pred_array[:, i]
        
        submission_df = pd.DataFrame(submission_data)
        
        # Save corrected submission
        output_file = f'output/submission_{model_name}_corrected.csv'
        submission_df.to_csv(output_file, index=False)
        
        print(f"✅ Saved corrected submission: {output_file}")
        print("Sample predictions:")
        print(submission_df.head())
        
        # Verify format matches sample
        print(f"Shape: {submission_df.shape} (should be (3, 6))")
        print(f"Columns: {list(submission_df.columns)}")

def create_baseline_submission():
    """Create submission from baseline model for comparison."""
    
    # Load baseline predictions if available
    baseline_file = Path('output/oof_preds_rf.csv')
    if baseline_file.exists():
        print("\n=== Creating Baseline Submission ===")
        
        # Load test data
        test_df = pd.read_csv('test.csv')
        test_ids = test_df['id'].values
        
        # Load baseline model
        baseline_model_path = Path('models/baseline_rf_multi.pkl')
        if baseline_model_path.exists():
            with open(baseline_model_path, 'rb') as f:
                baseline_model = pickle.load(f)
            
            # Load and process test features
            import sys
            sys.path.append('src')
            from utils.data_processing import DataProcessor
            
            data_processor = DataProcessor()
            X_test, _ = data_processor.prepare_data(test_df, is_training=False)
            
            # Make predictions
            baseline_preds = baseline_model.predict(X_test)
            
            # Create submission
            target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            submission_data = {'id': test_ids}
            
            for i, col in enumerate(target_columns):
                submission_data[col] = baseline_preds[:, i]
            
            submission_df = pd.DataFrame(submission_data)
            
            # Save baseline submission
            output_file = 'output/submission_baseline_corrected.csv'
            submission_df.to_csv(output_file, index=False)
            
            print(f"✅ Saved baseline submission: {output_file}")
            print("Sample predictions:")
            print(submission_df.head())

def main():
    """Main function to fix all submissions."""
    print("🔧 Fixing Submission Formats")
    print("=" * 50)
    
    # Fix advanced model submissions
    fix_submission_format()
    
    # Create baseline submission
    create_baseline_submission()
    
    print("\n" + "=" * 50)
    print("✅ All submissions fixed!")
    print("📁 Check output/ directory for *_corrected.csv files")

if __name__ == "__main__":
    main() 