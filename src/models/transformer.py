"""
Transformer-based models for molecular property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import pickle
import os


class SMILESTokenizer:
    """Custom tokenizer for SMILES strings."""
    
    def __init__(self):
        # Common SMILES tokens
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            'C': 4, 'c': 5, 'N': 6, 'n': 7, 'O': 8, 'o': 9,
            'S': 10, 's': 11, 'P': 12, 'p': 13, 'F': 14, 'Cl': 15,
            'Br': 16, 'I': 17, 'H': 18, 'B': 19, 'Si': 20,
            '(': 21, ')': 22, '[': 23, ']': 24, '=': 25, '#': 26,
            '-': 27, '+': 28, '@': 29, '/': 30, '\\': 31, '%': 32,
            '1': 33, '2': 34, '3': 35, '4': 36, '5': 37, '6': 38,
            '7': 39, '8': 40, '9': 41, '0': 42
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = 256
    
    def tokenize(self, smiles):
        """Tokenize SMILES string."""
        tokens = []
        i = 0
        while i < len(smiles):
            if i < len(smiles) - 1 and smiles[i:i+2] in self.vocab:
                tokens.append(smiles[i:i+2])
                i += 2
            elif smiles[i] in self.vocab:
                tokens.append(smiles[i])
                i += 1
            else:
                tokens.append('[UNK]')
                i += 1
        return tokens
    
    def encode(self, smiles, add_special_tokens=True):
        """Encode SMILES to token IDs."""
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.vocab['[PAD]']] * (self.max_length - len(token_ids)))
        
        # Create attention mask
        attention_mask = [1 if token_id != self.vocab['[PAD]'] else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


class SMILESDataset(Dataset):
    """Dataset for SMILES sequences and targets."""
    
    def __init__(self, smiles_list, targets, tokenizer):
        self.smiles_list = smiles_list
        self.targets = targets
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx] if self.targets is not None else None
        
        encoded = self.tokenizer.encode(smiles)
        
        item = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        if target is not None:
            item['targets'] = torch.tensor(target, dtype=torch.float32)
        
        return item


class TransformerEncoder(nn.Module):
    """Custom transformer encoder for SMILES."""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(256, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.embedding(input_ids) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.dropout(x)
        
        # Create padding mask
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling (excluding padding)
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        
        return x


class MolecularTransformer(nn.Module):
    """Transformer model for molecular property prediction."""
    
    def __init__(self, vocab_size, num_targets=5, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_layers, dropout)
        
        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleDict({
            'Tg': self._create_head(d_model),
            'FFV': self._create_head(d_model),
            'Tc': self._create_head(d_model),
            'Density': self._create_head(d_model),
            'Rg': self._create_head(d_model)
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_head(self, d_model):
        """Create prediction head for a specific target."""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask, targets=None):
        # Encode SMILES
        encoded = self.encoder(input_ids, attention_mask)
        encoded = self.dropout(encoded)
        
        # Multi-task predictions
        predictions = {}
        for target_name, head in self.prediction_heads.items():
            predictions[target_name] = head(encoded).squeeze(-1)
        
        if targets is not None:
            # Calculate loss
            losses = {}
            total_loss = 0
            
            target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
            for i, target_name in enumerate(target_names):
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


class TransformerRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible transformer regressor."""
    
    def __init__(self, d_model=256, nhead=8, num_layers=6, dropout=0.1, 
                 learning_rate=1e-4, batch_size=32, epochs=100, device=None):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = SMILESTokenizer()
        self.target_scaler = StandardScaler()
        self.model = None
        self.target_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    def fit(self, X, y):
        """Fit the transformer model."""
        print(f"Training transformer model on {self.device}...")
        
        # Scale targets
        y_scaled = self.target_scaler.fit_transform(y)
        
        # Create dataset
        dataset = SMILESDataset(X, y_scaled, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        vocab_size = len(self.tokenizer.vocab)
        self.model = MolecularTransformer(
            vocab_size=vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
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
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    optimizer.zero_grad()
                    predictions, loss, _ = self.model(input_ids, attention_mask, targets)
                    
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
        
        # Create dataset without targets
        dataset = SMILESDataset(X, None, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                pred_dict = self.model(input_ids, attention_mask)
                
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
            'tokenizer': self.tokenizer,
            'hyperparameters': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.d_model = hyperparams['d_model']
        self.nhead = hyperparams['nhead']
        self.num_layers = hyperparams['num_layers']
        self.dropout = hyperparams['dropout']
        
        # Restore tokenizer and scaler
        self.tokenizer = checkpoint['tokenizer']
        self.target_scaler = checkpoint['target_scaler']
        
        # Initialize and load model
        vocab_size = len(self.tokenizer.vocab)
        self.model = MolecularTransformer(
            vocab_size=vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self 