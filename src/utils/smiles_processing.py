#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SMILES processing utilities for polymer property prediction."""

import re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define atom types and bond types for feature extraction
ATOM_TYPES = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'B', '*']
BOND_TYPES = ['-', '=', '#', ':']


def canonicalize_smiles(smiles: str) -> str:
    """
    Convert SMILES to canonical form.
    
    Without RDKit, we can only perform basic cleaning.
    For production use, this should be replaced with RDKit's canonicalization.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Cleaned SMILES string
    """
    # Basic cleaning
    smiles = smiles.strip()
    
    return smiles


def augment_smiles(smiles: str, n_augmentations: int = 5) -> List[str]:
    """
    Generate augmented SMILES by performing simple transformations.
    
    Without RDKit, we can't generate valid SMILES with different atom orderings.
    This is a placeholder that returns the original SMILES.
    
    Args:
        smiles: SMILES string
        n_augmentations: Number of augmentations to generate
        
    Returns:
        List of augmented SMILES
    """
    # Return original SMILES n times - this should be replaced with proper augmentation
    return [smiles] * n_augmentations


def tokenize_smiles(smiles: str, max_length: int = 300) -> List[str]:
    """
    Tokenize SMILES string into individual tokens.
    
    Args:
        smiles: SMILES string
        max_length: Maximum token sequence length
        
    Returns:
        List of tokens
    """
    # Pattern to match:
    # - Atoms: C, N, O, etc.
    # - Digits: 1, 2, 3, etc.
    # - Brackets and symbols: (, ), [, ], etc.
    pattern = r'(\[[^\]]+]|Br|Cl|[#%\(\)\+\-\.\/0-9:=>@BCFHINOPS\[\\\]\*a-z\|\~])'
    tokens = re.findall(pattern, smiles)
    
    # Pad or truncate to fixed length
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [''] * (max_length - len(tokens))
    
    return tokens


def smiles_to_features(smiles: str) -> Dict[str, Any]:
    """
    Extract simple molecular features from SMILES.
    
    This implements basic feature extraction without RDKit.
    For production, this should be replaced with RDKit-based descriptors.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular features
    """
    features = {}
    
    # Basic features
    features['length'] = len(smiles)
    features['n_atoms'] = sum(1 for c in smiles if c.isalpha() and c.isupper())
    features['n_carbons'] = sum(1 for c in re.findall(r'C[^l]|C$', smiles))
    features['n_oxygens'] = smiles.count('O')
    features['n_nitrogens'] = smiles.count('N')
    features['n_rings'] = smiles.count('c') // 6  # Rough approximation for aromatic rings
    features['n_double_bonds'] = smiles.count('=')
    features['n_triple_bonds'] = smiles.count('#')
    
    # Count atom types
    for atom in ATOM_TYPES:
        if atom == 'C':
            # Avoid counting chlorine as carbon
            features[f'count_{atom}'] = len(re.findall(r'C[^l]|C$', smiles))
        else:
            features[f'count_{atom}'] = smiles.count(atom)
    
    # Count bond types
    for bond in BOND_TYPES:
        features[f'count_{bond}'] = smiles.count(bond)
    
    # Calculate molecular complexity (very simplified)
    features['complexity'] = len(set(smiles))
    features['branching'] = smiles.count('(')
    
    # Functional groups (very simplified)
    features['has_OH'] = int('OH' in smiles)
    features['has_NH'] = int('NH' in smiles)
    features['has_CN'] = int('CN' in smiles)
    features['has_carbonyl'] = int('C=O' in smiles)
    
    return features


def extract_dataset_features(df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
    """
    Extract features from all SMILES in a dataframe.
    
    Args:
        df: DataFrame containing SMILES
        smiles_col: Name of column containing SMILES
        
    Returns:
        DataFrame with molecular features
    """
    features = []
    
    for smiles in df[smiles_col]:
        if pd.isna(smiles):
            # Handle missing SMILES
            features.append({})
        else:
            features.append(smiles_to_features(smiles))
    
    features_df = pd.DataFrame(features)
    
    # Handle missing features
    features_df = features_df.fillna(0)
    
    return features_df


def preprocess_features(train_features: pd.DataFrame, 
                        test_features: Optional[pd.DataFrame] = None,
                        scale: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[StandardScaler]]:
    """
    Preprocess molecular features.
    
    Args:
        train_features: Training features
        test_features: Test features (optional)
        scale: Whether to scale features
        
    Returns:
        Processed training features, processed test features, and scaler (if scaling)
    """
    # Replace infinite values
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    if test_features is not None:
        test_features = test_features.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    train_features = train_features.fillna(0)
    if test_features is not None:
        test_features = test_features.fillna(0)
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        train_features = pd.DataFrame(
            scaler.fit_transform(train_features),
            columns=train_features.columns
        )
        
        if test_features is not None:
            test_features = pd.DataFrame(
                scaler.transform(test_features),
                columns=test_features.columns
            )
    
    return train_features, test_features, scaler


def extract_molecular_features(smiles_list):
    """
    Extract molecular features from a list of SMILES strings.
    
    Args:
        smiles_list: List or array of SMILES strings
        
    Returns:
        numpy array of molecular features
    """
    features = []
    
    for smiles in smiles_list:
        if pd.isna(smiles) or smiles == '':
            # Handle missing SMILES with default features
            mol_features = smiles_to_features('C')  # Default to methane
        else:
            mol_features = smiles_to_features(smiles)
        
        # Convert to list of values
        feature_values = list(mol_features.values())
        features.append(feature_values)
    
    return np.array(features) 