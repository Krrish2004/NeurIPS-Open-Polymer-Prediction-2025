# NeurIPS Open Polymer Prediction 2025: Solution Explanation

## Overview

This document explains our approach to solving the NeurIPS Open Polymer Prediction 2025 competition, which involves predicting five polymer properties from SMILES string representations.

## Problem Understanding

### Task
Predict five polymer properties from SMILES (Simplified Molecular Input Line Entry System) strings:
- Tg (glass transition temperature)
- FFV (fractional free volume)
- Tc (thermal conductivity)
- Density
- Rg (radius of gyration)

### Evaluation
The competition uses a weighted Mean Absolute Error (wMAE) metric that accounts for:
- Different scales of each property
- Varying numbers of data points for each property
- Inverse square-root weighting favoring rare properties

## Our Solution Approach

### 1. Data Analysis and Preprocessing

**SMILES Processing**
- Canonicalization for consistency
- Data augmentation through SMILES randomization
- Converting SMILES to both sequence tokens and molecular graphs

**Exploratory Data Analysis**
- Understanding property distributions
- Identifying correlations between properties
- Checking for missing values and outliers
- Property range normalization

### 2. Model Architecture Design

We implemented a hybrid approach with two main model architectures:

**Transformer-based Model**
- Adapts RoBERTa architecture for molecular language modeling
- Uses a multi-layer transformer encoder to process SMILES as sequences
- Applies multi-task learning with property-specific heads
- Benefits from pre-training on large SMILES datasets

**Graph Neural Network Model**
- Converts SMILES to molecular graphs (atoms as nodes, bonds as edges)
- Uses Graph Convolutional Networks (GCNs) to learn from molecular structure
- Incorporates atom features and bond types
- Applies global pooling to get molecule-level embeddings

**Ensemble Strategy**
- Weighted averaging of predictions from both models
- Optimizing weights based on validation performance
- Handling model uncertainty through variance estimation

### 3. Training Methodology

**Loss Function**
- Implementation of the weighted MAE directly as training objective
- Property-specific loss weighting based on competition formula
- Handling missing values during training

**Optimization**
- AdamW optimizer with weight decay for regularization
- Learning rate scheduling with warm-up and decay
- Gradient clipping to stabilize training
- Early stopping based on validation loss

**Cross-validation**
- 5-fold cross-validation to ensure robustness
- Hyperparameter optimization using validation performance

### 4. Implementation Details

**Framework and Libraries**
- PyTorch for model implementation and training
- RDKit for molecular processing and feature extraction
- DeepChem for advanced molecular representations
- PyTorch Geometric for graph neural networks

**Computational Resources**
- Training on a single GPU
- Mixed precision training for efficiency
- Optimized batch size for memory constraints

### 5. Results and Analysis

**Performance Metrics**
- Tracking per-property MAE to identify strengths/weaknesses
- Monitoring property-specific performance
- Analyzing prediction errors to guide model improvements

**Ablation Studies**
- Comparing transformer vs. graph-based approaches
- Evaluating impact of different featurization methods
- Measuring effectiveness of ensemble strategies

## Insights and Challenges

**Key Insights**
- Polymer properties are influenced by both local and global molecular features
- Multitask learning enables better generalization across properties
- SMILES representation captures important structural information

**Challenges**
- Limited data for some properties
- Balancing performance across all five properties
- Extracting meaningful representations from SMILES strings
- Computational efficiency under competition constraints

## Future Improvements

With additional time or resources, we would explore:
- More sophisticated graph neural network architectures
- Self-supervised pre-training on polymer databases
- Integration of physical and chemical domain knowledge
- Enhanced feature engineering using polymer science principles

## Conclusion

Our solution leverages state-of-the-art deep learning techniques adapted for molecular property prediction. By combining transformer and graph-based approaches, we created a robust multi-task learning framework that effectively handles the challenges of polymer property prediction. 