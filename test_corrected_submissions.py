#!/usr/bin/env python3
"""
Test the corrected submission generation
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def test_submission_format():
    """Test if the corrected submission format works."""
    
    print("🧪 Testing Corrected Submission Format")
    print("=" * 40)
    
    # Load test data
    test_df = pd.read_csv('test.csv')
    test_ids = test_df['id'].values
    print(f"Test IDs: {test_ids}")
    
    # Load existing predictions
    predictions_file = Path('output/advanced_predictions.pkl')
    if predictions_file.exists():
        with open(predictions_file, 'rb') as f:
            data = pickle.load(f)
            test_predictions = data.get('test_predictions', {})
        
        # Test the corrected format logic for each model
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        for model_name, predictions in test_predictions.items():
            print(f"\n--- Testing {model_name} ---")
            
            # Convert to numpy array
            test_pred = np.array(predictions)
            print(f"Original shape: {test_pred.shape}")
            
            # Apply the same logic as in the corrected train_advanced.py
            if test_pred.ndim == 1:
                if len(test_pred) == len(test_ids) * 5:  # 5 targets per sample
                    test_pred = test_pred.reshape(len(test_ids), 5)
                    print(f"Reshaped to: {test_pred.shape}")
                else:
                    print(f"⚠️ Unexpected prediction length: {len(test_pred)}")
                    continue
            elif test_pred.ndim == 2:
                if test_pred.shape[0] != len(test_ids) or test_pred.shape[1] != 5:
                    print(f"⚠️ Unexpected prediction shape: {test_pred.shape}")
                    continue
                else:
                    print(f"Shape already correct: {test_pred.shape}")
            else:
                print(f"⚠️ Unexpected prediction dimensions: {test_pred.ndim}")
                continue
            
            # Create submission DataFrame
            submission_data = {'id': test_ids}
            for i, col in enumerate(target_columns):
                submission_data[col] = test_pred[:, i]
            
            submission_df = pd.DataFrame(submission_data)
            
            # Test save
            test_output_file = f'output/test_submission_{model_name}.csv'
            submission_df.to_csv(test_output_file, index=False)
            
            print(f"✅ Test submission saved: {test_output_file}")
            print("Sample format:")
            print(submission_df.head())
            print(f"Shape: {submission_df.shape} (should be (3, 6))")
            
            # Verify format matches sample
            sample_df = pd.read_csv('sample_submission.csv')
            if list(submission_df.columns) == list(sample_df.columns):
                print("✅ Column format matches sample submission")
            else:
                print(f"❌ Column mismatch: {list(submission_df.columns)} vs {list(sample_df.columns)}")
    
    else:
        print("❌ No predictions file found")

def main():
    """Main test function."""
    test_submission_format()
    
    print("\n" + "=" * 40)
    print("🎯 Test completed!")

if __name__ == "__main__":
    main() 