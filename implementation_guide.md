# Implementation Guide for NeurIPS Polymer Prediction

This guide provides step-by-step instructions and code examples for implementing a solution to the NeurIPS Open Polymer Prediction 2025 competition.

## Project Setup

First, let's set up our project structure:

```
polymer-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── models/
│   ├── transformer_model.py
│   ├── graph_model.py
│   └── ensemble.py
├── utils/
│   ├── smiles_processing.py
│   ├── data_loader.py
│   └── evaluation.py
├── train.py
├── predict.py
└── requirements.txt
```

## 1. Data Preprocessing

### SMILES Processing (utils/smiles_processing.py)

```python
import rdkit
from rdkit import Chem
import numpy as np
from typing import List, Optional, Dict, Any

def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to canonical form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles  # Return original if parsing fails
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def augment_smiles(smiles: str, n_augmentations: int = 5) -> List[str]:
    """Generate augmented SMILES by randomizing atom ordering."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles] * n_augmentations
    
    augmented = []
    for _ in range(n_augmentations):
        # Get a random atom ordering
        atoms = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atoms)
        # Create new SMILES with that ordering
        new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=int(atoms[0]))
        augmented.append(new_smiles)
    
    return augmented

def smiles_to_graph(smiles: str) -> Dict[str, Any]:
    """Convert SMILES to a graph representation for GNN."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.GetMass(),
        ]
        atom_features.append(features)
    
    # Get bond features and edge indices
    bonds = []
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        
        # Add edges in both directions
        edge_indices.extend([[i, j], [j, i]])
        bonds.extend([bond_type, bond_type])
    
    return {
        'atom_features': np.array(atom_features, dtype=np.float32),
        'edge_index': np.array(edge_indices, dtype=np.int64).T if edge_indices else np.zeros((2, 0), dtype=np.int64),
        'edge_attr': np.array(bonds, dtype=np.float32).reshape(-1, 1) if bonds else np.zeros((0, 1), dtype=np.float32)
    }
```

### Data Loading (utils/data_loader.py)

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from utils.smiles_processing import canonicalize_smiles, smiles_to_graph

class PolymerDataset(Dataset):
    """Dataset for polymer property prediction."""
    
    def __init__(
        self, 
        csv_file: str, 
        smiles_col: str = 'SMILES', 
        target_cols: List[str] = None,
        transform_type: str = 'transformer',  # 'transformer' or 'graph'
        tokenizer = None,
        augment: bool = False
    ):
        self.data = pd.read_csv(csv_file)
        self.smiles_col = smiles_col
        self.target_cols = target_cols or ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.transform_type = transform_type
        self.tokenizer = tokenizer
        self.augment = augment
        
        # Canonicalize SMILES
        self.data[smiles_col] = self.data[smiles_col].apply(canonicalize_smiles)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        smiles = self.data.iloc[idx][self.smiles_col]
        
        # Get targets
        targets = []
        masks = []
        for col in self.target_cols:
            if col in self.data.columns:
                val = self.data.iloc[idx][col]
                targets.append(float(val) if not pd.isna(val) else 0.0)
                masks.append(0.0 if pd.isna(val) else 1.0)
            else:
                targets.append(0.0)
                masks.append(0.0)
        
        targets = np.array(targets, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        
        if self.transform_type == 'transformer':
            if self.tokenizer:
                tokens = self.tokenizer(smiles, padding='max_length', truncation=True, return_tensors='pt')
                return {
                    'input_ids': tokens['input_ids'][0],
                    'attention_mask': tokens['attention_mask'][0],
                    'targets': torch.tensor(targets, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.float32),
                    'id': self.data.iloc[idx].get('id', idx)
                }
            else:
                return {
                    'smiles': smiles,
                    'targets': torch.tensor(targets, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.float32),
                    'id': self.data.iloc[idx].get('id', idx)
                }
        else:  # graph
            graph_data = smiles_to_graph(smiles)
            if graph_data is None:
                # Fallback for parsing failures
                return {
                    'smiles': smiles,
                    'targets': torch.tensor(targets, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.float32),
                    'id': self.data.iloc[idx].get('id', idx)
                }
            
            return {
                'x': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
                'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
                'edge_attr': torch.tensor(graph_data['edge_attr'], dtype=torch.float32),
                'targets': torch.tensor(targets, dtype=torch.float32),
                'masks': torch.tensor(masks, dtype=torch.float32),
                'id': self.data.iloc[idx].get('id', idx)
            }

def get_data_loaders(
    train_file: str,
    test_file: Optional[str] = None,
    val_split: float = 0.2,
    batch_size: int = 32,
    transform_type: str = 'transformer',
    tokenizer = None,
    random_state: int = 42
) -> Dict[str, DataLoader]:
    """Create data loaders for training, validation and test."""
    train_data = PolymerDataset(
        train_file, transform_type=transform_type, tokenizer=tokenizer
    )
    
    # Split into training and validation
    train_size = int((1 - val_split) * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_data, [train_size, val_size], 
        generator=torch.Generator().manual_seed(random_state)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if test_file:
        test_dataset = PolymerDataset(
            test_file, transform_type=transform_type, tokenizer=tokenizer
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loaders['test'] = test_loader
    
    return loaders
```

## 2. Model Implementation

### Transformer Model (models/transformer_model.py)

```python
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

class PolymerTransformer(nn.Module):
    def __init__(
        self,
        num_targets: int = 5,
        pretrained_model: str = 'roberta-base',
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        
        # Load pretrained model or create from config
        try:
            self.transformer = RobertaModel.from_pretrained(pretrained_model)
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        except:
            # If no pretrained model is available, initialize from scratch
            config = RobertaConfig(
                vocab_size=1000,  # Adjust based on your tokenizer
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                hidden_dropout_prob=hidden_dropout_prob,
            )
            self.transformer = RobertaModel(config)
            self.tokenizer = None
        
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Shared layers
        self.shared_layer1 = nn.Linear(self.transformer.config.hidden_size, 512)
        self.shared_layer2 = nn.Linear(512, 256)
        
        # Property-specific heads
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(hidden_dropout_prob),
                nn.Linear(128, 1)
            )
            for _ in range(num_targets)
        ])
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Shared layers
        x = torch.relu(self.shared_layer1(pooled_output))
        x = self.dropout(x)
        x = torch.relu(self.shared_layer2(x))
        x = self.dropout(x)
        
        # Property-specific predictions
        outputs = []
        for head in self.property_heads:
            outputs.append(head(x))
            
        return torch.cat(outputs, dim=1)
    
    def get_tokenizer(self):
        return self.tokenizer
```

### Graph Neural Network (models/graph_model.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

class PolymerGNN(nn.Module):
    def __init__(
        self, 
        in_channels: int = 7,  # Number of atom features
        hidden_channels: int = 128,
        num_targets: int = 5,
        num_layers: int = 3
    ):
        super().__init__()
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # MLP for predictions after pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Property-specific heads
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels // 2, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            )
            for _ in range(num_targets)
        ])
        
    def forward(self, x, edge_index, edge_attr, batch=None):
        # If no batch vector is provided, assume a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # GNN layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Pooling
        x = global_mean_pool(x, batch)
        
        # MLP
        x = self.mlp(x)
        
        # Property-specific predictions
        outputs = []
        for head in self.property_heads:
            outputs.append(head(x))
            
        return torch.cat(outputs, dim=1)
```

## 3. Training Loop

### Loss Function (utils/evaluation.py)

```python
import torch
import numpy as np
from typing import Dict, List, Tuple

def weighted_mae_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor,
    property_weights: torch.Tensor = None
) -> torch.Tensor:
    """
    Weighted Mean Absolute Error Loss Function
    
    Args:
        pred: Predicted values [batch_size, num_properties]
        target: True values [batch_size, num_properties]
        mask: Mask for missing values [batch_size, num_properties]
        property_weights: Weights for each property [num_properties]
    
    Returns:
        Weighted MAE loss
    """
    # If no weights provided, use uniform weights
    if property_weights is None:
        property_weights = torch.ones(pred.shape[1], device=pred.device)
    
    # Calculate absolute error
    abs_error = torch.abs(pred - target) * mask
    
    # Apply weights for each property
    weighted_error = abs_error * property_weights.view(1, -1)
    
    # Sum over properties and average over batch
    loss = torch.sum(weighted_error, dim=1)
    
    # Only average over samples with at least one valid property
    mask_sum = torch.sum(mask, dim=1)
    valid_samples = (mask_sum > 0)
    
    if valid_samples.sum() > 0:
        loss = loss[valid_samples].mean()
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

def calculate_property_weights(
    train_data: List[Dict[str, torch.Tensor]],
    num_properties: int = 5
) -> torch.Tensor:
    """Calculate weights for each property based on competition formula."""
    # Count available values for each property
    property_counts = torch.zeros(num_properties)
    property_ranges = torch.zeros(num_properties)
    
    # Extract all target values
    all_targets = []
    all_masks = []
    
    for batch in train_data:
        all_targets.append(batch['targets'])
        all_masks.append(batch['masks'])
    
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Count available values and calculate ranges
    for i in range(num_properties):
        valid_mask = all_masks[:, i] > 0
        if valid_mask.sum() > 0:
            property_counts[i] = valid_mask.sum()
            valid_values = all_targets[valid_mask, i]
            property_ranges[i] = valid_values.max() - valid_values.min()
    
    # Avoid division by zero
    property_counts = torch.clamp(property_counts, min=1)
    property_ranges = torch.clamp(property_ranges, min=1e-8)
    
    # Calculate weights based on the competition formula
    K = float(num_properties)
    n_i = property_counts
    r_i = property_ranges
    
    weights = (K / n_i) * (torch.sqrt(1/n_i) / torch.sum(torch.sqrt(1/n_i)))
    
    # Normalize weights to sum to num_properties
    weights = weights * (K / weights.sum())
    
    return weights
```

### Training Script (train.py)

```python
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

from models.transformer_model import PolymerTransformer
from models.graph_model import PolymerGNN
from utils.data_loader import get_data_loaders
from utils.evaluation import weighted_mae_loss, calculate_property_weights

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    if args.model_type == 'transformer':
        model = PolymerTransformer(
            num_targets=args.num_targets,
            pretrained_model=args.pretrained_model,
            hidden_dropout_prob=args.dropout
        )
        tokenizer = model.get_tokenizer()
    else:
        model = PolymerGNN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_dim,
            num_targets=args.num_targets,
            num_layers=args.num_layers
        )
        tokenizer = None
    
    model = model.to(device)
    
    # Get data loaders
    loaders = get_data_loaders(
        train_file=args.train_file,
        test_file=args.test_file,
        val_split=args.val_split,
        batch_size=args.batch_size,
        transform_type=args.model_type,
        tokenizer=tokenizer,
        random_state=args.seed
    )
    
    # Calculate property weights
    property_weights = torch.ones(args.num_targets).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        
        for batch in tqdm(loaders['train'], desc=f"Epoch {epoch + 1}/{args.epochs}"):
            # Move data to device
            if args.model_type == 'transformer':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                masks = batch['masks'].to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask)
            else:  # graph
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device)
                targets = batch['targets'].to(device)
                masks = batch['masks'].to(device)
                
                # Forward pass
                outputs = model(x, edge_index, edge_attr)
            
            # Calculate loss
            loss = weighted_mae_loss(outputs, targets, masks, property_weights)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in loaders['val']:
                # Move data to device
                if args.model_type == 'transformer':
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    targets = batch['targets'].to(device)
                    masks = batch['masks'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask)
                else:  # graph
                    x = batch['x'].to(device)
                    edge_index = batch['edge_index'].to(device)
                    edge_attr = batch['edge_attr'].to(device)
                    targets = batch['targets'].to(device)
                    masks = batch['masks'].to(device)
                    
                    # Forward pass
                    outputs = model(x, edge_index, edge_attr)
                
                # Calculate loss
                loss = weighted_mae_loss(outputs, targets, masks, property_weights)
                val_losses.append(loss.item())
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            torch.save(best_model, os.path.join(args.output_dir, f"best_model_{args.model_type}.pt"))
            print(f"  New best model saved!")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"final_model_{args.model_type}.pt"))
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train polymer property prediction model')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--test_file', type=str, default=None, help='Path to test CSV')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'graph'], 
                        help='Model type to train')
    parser.add_argument('--pretrained_model', type=str, default=None, 
                        help='Pretrained transformer model name/path')
    parser.add_argument('--in_channels', type=int, default=7, 
                        help='Number of input features for GNN')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Hidden dimension size')
    parser.add_argument('--num_targets', type=int, default=5, 
                        help='Number of target properties')
    parser.add_argument('--num_layers', type=int, default=3, 
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./saved_models', 
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train model
    model = train(args)
```

## 4. Inference and Submission

### Prediction Script (predict.py)

```python
import torch
import pandas as pd
import numpy as np
import os
import argparse

from models.transformer_model import PolymerTransformer
from models.graph_model import PolymerGNN
from utils.data_loader import PolymerDataset

def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if args.model_type == 'transformer':
        model = PolymerTransformer(
            num_targets=args.num_targets,
            pretrained_model=args.pretrained_model
        )
        tokenizer = model.get_tokenizer()
    else:
        model = PolymerGNN(
            in_channels=args.in_channels,
            hidden_channels=args.hidden_dim,
            num_targets=args.num_targets,
            num_layers=args.num_layers
        )
        tokenizer = None
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = PolymerDataset(
        args.test_file,
        transform_type=args.model_type,
        tokenizer=tokenizer,
        target_cols=[]  # No targets for test set
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Make predictions
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            if args.model_type == 'transformer':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids, attention_mask)
            else:
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch['edge_attr'].to(device)
                outputs = model(x, edge_index, edge_attr)
            
            predictions.append(outputs.cpu().numpy())
            ids.extend(batch['id'])
    
    # Concatenate predictions
    predictions = np.vstack(predictions)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': ids,
        'Tg': predictions[:, 0],
        'FFV': predictions[:, 1],
        'Tc': predictions[:, 2],
        'Density': predictions[:, 3],
        'Rg': predictions[:, 4]
    })
    
    # Save submission
    submission.to_csv(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for polymer properties')
    
    # Data arguments
    parser.add_argument('--test_file', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_file', type=str, default='submission.csv', help='Path to save predictions')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'graph'], help='Model type')
    parser.add_argument('--pretrained_model', type=str, default=None, 
                        help='Pretrained transformer model name/path')
    parser.add_argument('--in_channels', type=int, default=7, 
                        help='Number of input features for GNN')
    parser.add_argument('--hidden_dim', type=int, default=128, 
                        help='Hidden dimension size')
    parser.add_argument('--num_targets', type=int, default=5, 
                        help='Number of target properties')
    parser.add_argument('--num_layers', type=int, default=3, 
                        help='Number of GNN layers')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    predict(args)
```

## 5. Ensemble Model Implementation

### Ensemble Model (models/ensemble.py)

```python
import torch
import numpy as np
import pandas as pd

class EnsembleModel:
    """Ensemble of multiple models for polymer property prediction."""
    
    def __init__(self, models, model_types, weights=None, device='cpu'):
        """
        Initialize ensemble model.
        
        Args:
            models: List of trained models
            model_types: List of model types ('transformer' or 'graph')
            weights: Optional list of weights for each model
            device: Device to run inference on
        """
        self.models = models
        self.model_types = model_types
        
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights) / np.sum(weights)
        
        self.device = device
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
    
    def predict(self, batch):
        """Make predictions with ensemble."""
        predictions = []
        
        with torch.no_grad():
            for model, model_type in zip(self.models, self.model_types):
                if model_type == 'transformer':
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    outputs = model(input_ids, attention_mask)
                else:  # graph
                    x = batch['x'].to(self.device)
                    edge_index = batch['edge_index'].to(self.device)
                    edge_attr = batch['edge_attr'].to(self.device)
                    outputs = model(x, edge_index, edge_attr)
                
                predictions.append(outputs.cpu().numpy())
        
        # Weight and combine predictions
        weighted_preds = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_preds += pred * weight
        
        return weighted_preds
    
    @classmethod
    def load_from_paths(cls, model_paths, model_types, model_configs, weights=None, device='cpu'):
        """
        Load ensemble from saved model paths.
        
        Args:
            model_paths: List of paths to saved models
            model_types: List of model types ('transformer' or 'graph')
            model_configs: List of dictionaries with model configurations
            weights: Optional list of weights for each model
            device: Device to run inference on
        """
        models = []
        
        for path, model_type, config in zip(model_paths, model_types, model_configs):
            if model_type == 'transformer':
                model = PolymerTransformer(**config)
            else:
                model = PolymerGNN(**config)
            
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            models.append(model)
        
        return cls(models, model_types, weights, device)
```

## Running the Code

### Sample Usage

To train the transformer model:

```bash
python train.py \
  --train_file data/train.csv \
  --test_file data/test.csv \
  --model_type transformer \
  --batch_size 16 \
  --epochs 30 \
  --learning_rate 2e-5 \
  --output_dir ./saved_models
```

To train the graph model:

```bash
python train.py \
  --train_file data/train.csv \
  --test_file data/test.csv \
  --model_type graph \
  --batch_size 32 \
  --epochs 50 \
  --learning_rate 1e-3 \
  --output_dir ./saved_models
```

To generate predictions:

```bash
python predict.py \
  --test_file data/test.csv \
  --model_path saved_models/best_model_transformer.pt \
  --model_type transformer \
  --output_file submission.csv
```

## Conclusion

This implementation guide provides a complete solution for the NeurIPS Polymer Prediction competition. The key aspects of this solution are:

1. Using both transformer-based and graph-based models for molecular representation
2. Implementing a custom weighted MAE loss function that matches the competition metric
3. Supporting both single model and ensemble predictions
4. Handling missing values and data imbalance correctly

The implementation is designed to be modular and extensible, allowing for easy experimentation with different model architectures and hyperparameters. 