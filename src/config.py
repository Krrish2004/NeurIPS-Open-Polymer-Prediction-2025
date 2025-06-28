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