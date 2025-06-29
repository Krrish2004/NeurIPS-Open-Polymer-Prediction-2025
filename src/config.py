#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Configuration settings for the polymer prediction project."""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / 'data'
OUTPUT_DIR = ROOT_DIR / 'output'
MODEL_DIR = ROOT_DIR / 'models'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data files
TRAIN_FILE = ROOT_DIR / 'train.csv'
TEST_FILE = ROOT_DIR / 'test.csv'
SAMPLE_SUBMISSION_FILE = ROOT_DIR / 'sample_submission.csv'

# Model parameters
RANDOM_STATE = 42
TARGET_COLUMNS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
N_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Transformer model settings
TRANSFORMER_CONFIG = {
    'hidden_size': 768,
    'num_hidden_layers': 6,
    'num_attention_heads': 12,
    'hidden_dropout_prob': 0.1,
}

# GNN model settings
GNN_CONFIG = {
    'in_channels': 7,  # Number of atom features
    'hidden_channels': 128,
    'num_layers': 3,
}

# Feature extraction parameters
MAX_SMILES_LENGTH = 300
FEATURE_SCALING = True

# Evaluation
EVALUATION_METRIC = 'weighted_mae'


class Config:
    """Configuration class for the polymer prediction project."""
    
    def __init__(self):
        # Paths
        self.root_dir = ROOT_DIR
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.models_dir = MODEL_DIR
        
        # Data files
        self.train_path = TRAIN_FILE
        self.test_path = TEST_FILE
        self.sample_submission_path = SAMPLE_SUBMISSION_FILE
        
        # Model parameters
        self.random_state = RANDOM_STATE
        self.target_columns = TARGET_COLUMNS
        self.cv_folds = N_FOLDS
        self.batch_size = BATCH_SIZE
        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.weight_decay = WEIGHT_DECAY
        self.early_stopping_patience = EARLY_STOPPING_PATIENCE
        
        # Feature extraction
        self.max_smiles_length = MAX_SMILES_LENGTH
        self.feature_scaling = FEATURE_SCALING
        
        # Evaluation
        self.evaluation_metric = EVALUATION_METRIC
        
        # Property weights for weighted MAE calculation
        self.property_weights = {
            'Tg': 0.15,
            'FFV': 0.2, 
            'Tc': 0.2,
            'Density': 0.2,
            'Rg': 0.25
        }
        
        # Model configurations
        self.transformer_config = TRANSFORMER_CONFIG
        self.gnn_config = GNN_CONFIG 