# NeurIPS Open Polymer Prediction 2025: Architecture Design

## Problem Overview

This competition challenges participants to predict five polymer properties from SMILES string representations:
- Tg (glass transition temperature)
- FFV (fractional free volume)
- Tc (thermal conductivity)
- Density
- Rg (radius of gyration)

The evaluation metric is a weighted Mean Absolute Error (wMAE) that accounts for different value ranges and data availability across properties.

## Architecture Design

### 1. Model Architecture

We'll implement a multitask learning approach with the following components:

```
SMILES Input → Molecular Representation → Shared Encoder → Property-Specific Heads → 5 Properties
```

#### Representation Options:

1. **Transformer-Based Approach**:
   - SMILES tokenization
   - Pre-trained polymer language model (e.g., modified RoBERTa)
   - Fine-tuning on our specific task

2. **Graph Neural Network Approach**:
   - Convert SMILES to molecular graphs
   - Message passing neural networks
   - Readout functions to generate molecule-level embeddings

3. **Ensemble Approach**:
   - Combine predictions from multiple models
   - Weight ensemble predictions based on validation performance

### 2. Loss Function

We'll implement the competition's weighted MAE directly as our training objective:

```
wMAE = (1/|X|) * ∑ ∑ w_i · |ŷ_i(X) - y_i(X)|
```

Where `w_i` accounts for property scale and data frequency.

### 3. Training Strategy

- **Data Splits**: 80% training, 20% validation
- **Cross-Validation**: 5-fold CV to ensure robust evaluation
- **Learning Rate**: Cyclic learning rate with warm-up
- **Regularization**: Dropout, weight decay, and early stopping

### 4. Inference Pipeline

1. Preprocess SMILES strings (canonicalization)
2. Generate molecular representations
3. Forward pass through model
4. Post-process predictions (scaling/clipping if necessary)
5. Output final CSV in required format

## Technology Stack

- **Framework**: PyTorch
- **Chemistry Libraries**: RDKit, DeepChem
- **Training**: Single GPU training with mixed precision
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Local CSV logging (within competition constraints) 