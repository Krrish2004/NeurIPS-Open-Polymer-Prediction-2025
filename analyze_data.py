#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
print("Loading datasets...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Basic information
print("\nTrain dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
print("Sample submission shape:", sample_submission.shape)

# Check missing values
print("\nMissing values in train dataset:")
missing_train = train_df.isnull().sum()
print(missing_train)
missing_percentage = (missing_train / len(train_df)) * 100
print("\nMissing percentage:")
print(missing_percentage)

# Data statistics for each property
print("\nStatistics for each property:")
stats = train_df.describe()
print(stats)

# Check SMILES length distribution
train_df['smiles_length'] = train_df['SMILES'].str.len()
print("\nSMILES length statistics:")
print(train_df['smiles_length'].describe())

# Count occurrences of each element in SMILES
print("\nAnalyzing SMILES patterns...")
common_elements = {}
for smiles in train_df['SMILES'].dropna().values[:100]:  # Sample first 100 for speed
    # Count atoms and bonds
    for char in smiles:
        if char.isalpha():
            if char in common_elements:
                common_elements[char] += 1
            else:
                common_elements[char] = 1
                
# Convert to Series for easy display
element_counts = pd.Series(common_elements).sort_values(ascending=False)
print("Most common elements in SMILES:")
print(element_counts.head(10))

# Check correlations between properties
print("\nCorrelation between properties:")
correlation = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].corr()
print(correlation)

# Save correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Between Polymer Properties')
plt.savefig('property_correlation.png')
print("Correlation heatmap saved as 'property_correlation.png'")

# Save property distributions
plt.figure(figsize=(15, 10))
for i, col in enumerate(['Tg', 'FFV', 'Tc', 'Density', 'Rg']):
    plt.subplot(2, 3, i+1)
    sns.histplot(train_df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.savefig('property_distributions.png')
print("Property distributions saved as 'property_distributions.png'")

# Analysis of data availability
plt.figure(figsize=(10, 6))
missing = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].isnull().sum()
plt.bar(missing.index, missing.values)
plt.title('Missing Values Count by Property')
plt.ylabel('Number of Missing Values')
plt.savefig('missing_values.png')
print("Missing values chart saved as 'missing_values.png'")

print("\nAnalysis complete!") 