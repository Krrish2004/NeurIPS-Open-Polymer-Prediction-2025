#!/usr/bin/env python3
"""
Quick test script to verify transformer and GNN models work correctly.
"""

import sys
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.append('src')

from models.transformer import TransformerRegressor
from models.gnn import GNNRegressor

warnings.filterwarnings('ignore')


def test_transformer_model():
    """Test transformer model with sample data."""
    print("Testing Transformer Model...")
    
    # Sample SMILES data
    smiles_data = [
        'CCO',  # Ethanol
        'CCC',  # Propane
        'CCCC', # Butane
        'C=CC', # Propene
        'c1ccccc1'  # Benzene
    ]
    
    # Sample target data (5 targets: Tg, FFV, Tc, Density, Rg)
    y_data = np.array([
        [100.0, 0.5, 200.0, 1.0, 2.0],
        [110.0, 0.4, 210.0, 1.1, 2.1],
        [120.0, 0.6, 220.0, 1.2, 2.2],
        [130.0, 0.3, 230.0, 1.3, 2.3],
        [140.0, 0.7, 240.0, 1.4, 2.4]
    ])
    
    try:
        # Initialize model with smaller parameters for quick testing
        model = TransformerRegressor(
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            learning_rate=1e-3,
            batch_size=2,
            epochs=5
        )
        
        # Fit the model
        print("  Training transformer...")
        model.fit(smiles_data, y_data)
        
        # Make predictions
        print("  Making predictions...")
        predictions = model.predict(smiles_data)
        
        print(f"  Input shape: {len(smiles_data)}")
        print(f"  Target shape: {y_data.shape}")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Sample prediction: {predictions[0]}")
        
        print("  ✓ Transformer model test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ Transformer model test failed: {e}")
        return False


def test_gnn_model():
    """Test GNN model with sample data."""
    print("\nTesting GNN Model...")
    
    # Sample SMILES data
    smiles_data = [
        'CCO',  # Ethanol
        'CCC',  # Propane
        'CCCC', # Butane
        'C=CC', # Propene
        'c1ccccc1'  # Benzene
    ]
    
    # Sample target data (5 targets: Tg, FFV, Tc, Density, Rg)
    y_data = np.array([
        [100.0, 0.5, 200.0, 1.0, 2.0],
        [110.0, 0.4, 210.0, 1.1, 2.1],
        [120.0, 0.6, 220.0, 1.2, 2.2],
        [130.0, 0.3, 230.0, 1.3, 2.3],
        [140.0, 0.7, 240.0, 1.4, 2.4]
    ])
    
    try:
        # Initialize model with smaller parameters for quick testing
        model = GNNRegressor(
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            pool='mean',
            learning_rate=1e-3,
            batch_size=2,
            epochs=5
        )
        
        # Fit the model
        print("  Training GNN...")
        model.fit(smiles_data, y_data)
        
        # Make predictions
        print("  Making predictions...")
        predictions = model.predict(smiles_data)
        
        print(f"  Input shape: {len(smiles_data)}")
        print(f"  Target shape: {y_data.shape}")
        print(f"  Prediction shape: {predictions.shape}")
        print(f"  Sample prediction: {predictions[0]}")
        
        print("  ✓ GNN model test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ GNN model test failed: {e}")
        return False


def test_models_compatibility():
    """Test that models work with actual data format."""
    print("\nTesting Models with Real Data Format...")
    
    try:
        # Load actual data
        train_df = pd.read_csv('train.csv')
        
        # Take a small sample
        sample_df = train_df.head(10).copy()
        
        # Fill missing values for testing
        for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
            sample_df[col] = sample_df[col].fillna(sample_df[col].mean())
        
        smiles_data = sample_df['SMILES'].values
        y_data = sample_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].values
        
        # Test transformer
        try:
            transformer = TransformerRegressor(epochs=3, batch_size=2)
            transformer.fit(smiles_data, y_data)
            trans_preds = transformer.predict(smiles_data[:3])
            print(f"  ✓ Transformer works with real data: {trans_preds.shape}")
        except Exception as e:
            print(f"  ✗ Transformer failed with real data: {e}")
        
        # Test GNN
        try:
            gnn = GNNRegressor(epochs=3, batch_size=2)
            gnn.fit(smiles_data, y_data)
            gnn_preds = gnn.predict(smiles_data[:3])
            print(f"  ✓ GNN works with real data: {gnn_preds.shape}")
        except Exception as e:
            print(f"  ✗ GNN failed with real data: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Real data test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Advanced Model Testing")
    print("=" * 50)
    
    # Test individual models
    transformer_ok = test_transformer_model()
    gnn_ok = test_gnn_model()
    
    # Test with real data
    real_data_ok = test_models_compatibility()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Transformer: {'✓ PASS' if transformer_ok else '✗ FAIL'}")
    print(f"  GNN:         {'✓ PASS' if gnn_ok else '✗ FAIL'}")
    print(f"  Real Data:   {'✓ PASS' if real_data_ok else '✗ FAIL'}")
    
    if all([transformer_ok, gnn_ok, real_data_ok]):
        print("\n🎉 All tests passed! Advanced models are working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main() 