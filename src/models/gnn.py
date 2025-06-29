"""
Graph Neural Network models for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import os
import re


class SMILESToGraph:
    """Convert SMILES to molecular graph without RDKit."""
    
    def __init__(self):
        # Atom types and their properties
        self.atom_types = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
            'B': 10, 'Si': 11, 'c': 12, 'n': 13, 'o': 14, 's': 15, 'p': 16
        }
        
        # Atom features (atomic number, valence, etc.)
        self.atom_features = {
            'C': [6, 4], 'N': [7, 3], 'O': [8, 2], 'S': [16, 2], 'P': [15, 3],
            'F': [9, 1], 'Cl': [17, 1], 'Br': [35, 1], 'I': [53, 1], 'H': [1, 1],
            'B': [5, 3], 'Si': [14, 4], 'c': [6, 4], 'n': [7, 3], 'o': [8, 2],
            's': [16, 2], 'p': [15, 3]
        }
        
        # Bond types
        self.bond_types = {'-': 0, '=': 1, '#': 2, ':': 3}
    
    def parse_smiles(self, smiles):
        """Parse SMILES string to extract atoms and bonds."""
        atoms = []
        bonds = []
        atom_map = {}  # Map position to atom index
        
        i = 0
        atom_idx = 0
        stack = []  # For handling branches
        
        while i < len(smiles):
            char = smiles[i]
            
            if char == '(':
                # Start of branch
                stack.append(atom_idx - 1)
                i += 1
            elif char == ')':
                # End of branch
                if stack:
                    stack.pop()
                i += 1
            elif char in ['[', ']']:
                # Skip bracket notations for now
                i += 1
            elif char.isdigit():
                # Ring closure
                ring_num = int(char)
                if ring_num in atom_map:
                    # Close ring
                    bonds.append((atom_map[ring_num], atom_idx - 1, 0))  # Single bond
                else:
                    # Open ring
                    atom_map[ring_num] = atom_idx - 1
                i += 1
            elif char in self.bond_types:
                # Bond type (will be applied to next connection)
                i += 1
            elif char in self.atom_types:
                # Atom
                atom_type = char
                
                # Check for two-character atoms (Cl, Br)
                if i + 1 < len(smiles) and smiles[i:i+2] in self.atom_types:
                    atom_type = smiles[i:i+2]
                    i += 1
                
                atoms.append(atom_type)
                
                # Add bond to previous atom (if not first atom and not after branch)
                if atom_idx > 0 and (not stack or atom_idx - 1 not in [s for s in stack]):
                    bonds.append((atom_idx - 1, atom_idx, 0))  # Default single bond
                
                # Add bonds from stack (branch connections)
                if stack:
                    bonds.append((stack[-1], atom_idx, 0))
                
                atom_idx += 1
                i += 1
            else:
                i += 1
        
        return atoms, bonds
    
    def create_node_features(self, atoms):
        """Create node feature matrix."""
        features = []
        for atom in atoms:
            # Basic features: one-hot encoding of atom type + atomic number + valence
            atom_feat = [0] * len(self.atom_types)
            if atom in self.atom_types:
                atom_feat[self.atom_types[atom]] = 1
            
            # Add atomic properties
            if atom in self.atom_features:
                atom_feat.extend(self.atom_features[atom])
            else:
                atom_feat.extend([0, 0])  # Default values
            
            features.append(atom_feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def create_edge_features(self, bonds, num_atoms):
        """Create edge index and edge features."""
        if not bonds:
            # Single atom molecule
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
            return edge_index, edge_attr
        
        edge_index = []
        edge_attr = []
        
        for bond in bonds:
            if len(bond) >= 2:
                src, dst = bond[0], bond[1]
                bond_type = bond[2] if len(bond) > 2 else 0
                
                # Add both directions (undirected graph)
                edge_index.extend([[src, dst], [dst, src]])
                edge_attr.extend([[bond_type], [bond_type]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def smiles_to_graph(self, smiles):
        """Convert SMILES to PyTorch Geometric Data object."""
        try:
            atoms, bonds = self.parse_smiles(smiles)
            
            if not atoms:
                # Empty molecule - create single carbon
                atoms = ['C']
                bonds = []
            
            # Create features
            x = self.create_node_features(atoms)
            edge_index, edge_attr = self.create_edge_features(bonds, len(atoms))
            
            # Create graph data
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            return data
        
        except Exception as e:
            # Fallback: create single carbon atom
            x = self.create_node_features(['C'])
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolecularGNN(nn.Module):
    """Graph Neural Network for molecular property prediction."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.1, pool='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.pool = pool
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict({
            'Tg': self._create_head(hidden_dim),
            'FFV': self._create_head(hidden_dim),
            'Tc': self._create_head(hidden_dim),
            'Density': self._create_head(hidden_dim),
            'Rg': self._create_head(hidden_dim)
        })
    
    def _create_head(self, hidden_dim):
        """Create prediction head for a specific target."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr, batch, targets=None):
        # Graph convolutions
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        if self.pool == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        
        # Multi-task predictions
        predictions = {}
        for target_name, head in self.prediction_heads.items():
            predictions[target_name] = head(x).squeeze(-1)
        
        if targets is not None:
            # Calculate loss
            losses = {}
            total_loss = 0
            
            target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            
            # Handle different target tensor shapes
            if targets.dim() == 1:
                # Single target or flattened targets
                targets = targets.view(-1, len(target_names))
            
            for i, target_name in enumerate(target_names):
                if i < targets.shape[1]:  # Check if target exists
                    target_vals = targets[:, i]
                    pred_vals = predictions[target_name]
                    
                    # Only calculate loss for non-NaN targets
                    mask = ~torch.isnan(target_vals)
                    if mask.sum() > 0:
                        loss = F.mse_loss(pred_vals[mask], target_vals[mask])
                        losses[target_name] = loss
                        total_loss += loss
            
            return predictions, total_loss, losses
        
        return predictions


class GNNRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible GNN regressor."""
    
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.1, pool='mean',
                 learning_rate=1e-3, batch_size=32, epochs=100, device=None):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.graph_converter = SMILESToGraph()
        self.target_scaler = StandardScaler()
        self.model = None
        self.input_dim = None
        self.target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def _create_graphs(self, smiles_list, targets=None):
        """Convert SMILES list to graph dataset."""
        graphs = []
        for i, smiles in enumerate(smiles_list):
            graph = self.graph_converter.smiles_to_graph(smiles)
            
            if targets is not None:
                target_vals = targets[i]
                if isinstance(target_vals, (list, tuple, np.ndarray)):
                    # Ensure we have exactly 5 targets
                    if len(target_vals) == 5:
                        graph.y = torch.tensor(target_vals, dtype=torch.float32)
                    else:
                        # Pad or truncate to 5 targets
                        padded_targets = np.full(5, np.nan)
                        padded_targets[:min(len(target_vals), 5)] = target_vals[:5]
                        graph.y = torch.tensor(padded_targets, dtype=torch.float32)
                else:
                    # Single target - pad to 5
                    padded_targets = np.full(5, np.nan)
                    padded_targets[0] = target_vals
                    graph.y = torch.tensor(padded_targets, dtype=torch.float32)
            
            graphs.append(graph)
        
        return graphs
    
    def fit(self, X, y):
        """Fit the GNN model."""
        print(f"Training GNN model on {self.device}...")
        
        # Scale targets
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Create graphs
        graphs = self._create_graphs(X, y_scaled)
        
        # Determine input dimension
        self.input_dim = graphs[0].x.shape[1]
        
        # Create data loader
        dataloader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = MolecularGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pool=self.pool
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.model.train()
        from tqdm import tqdm
        
        with tqdm(total=self.epochs, desc="Training", unit="epoch", leave=False) as epoch_pbar:
            for epoch in range(self.epochs):
                total_loss = 0
                num_batches = 0
                
                for batch in dataloader:
                    batch = batch.to(self.device)
                    
                    optimizer.zero_grad()
                    predictions, loss, _ = self.model(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                scheduler.step(avg_loss)
                
                epoch_pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})
                epoch_pbar.update(1)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        self.model.eval()
        predictions = []
        
        # Create graphs without targets
        graphs = self._create_graphs(X)
        dataloader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                
                pred_dict = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Stack predictions in correct order
                batch_preds = torch.stack([
                    pred_dict['Tg'],
                    pred_dict['FFV'], 
                    pred_dict['Tc'],
                    pred_dict['Density'],
                    pred_dict['Rg']
                ], dim=1)
                
                predictions.append(batch_preds.cpu().numpy())
        
        # Concatenate and inverse scale
        predictions = np.concatenate(predictions, axis=0)
        predictions = self.target_scaler.inverse_transform(predictions)
        
        return predictions
    
    def save(self, filepath):
        """Save the model."""
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_scaler': self.target_scaler,
            'input_dim': self.input_dim,
            'hyperparameters': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'pool': self.pool
            }
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.hidden_dim = hyperparams['hidden_dim']
        self.num_layers = hyperparams['num_layers']
        self.dropout = hyperparams['dropout']
        self.pool = hyperparams['pool']
        self.input_dim = checkpoint['input_dim']
        
        # Restore scaler
        self.target_scaler = checkpoint['target_scaler']
        
        # Initialize and load model
        self.model = MolecularGNN(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pool=self.pool
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self 