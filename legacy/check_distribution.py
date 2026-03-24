import pandas as pd
import numpy as np

# Load data (path updated for integrated-scripts directory)
data = pd.read_csv('../processed_data/integrated_features_improved.csv')

print("="*60)
print("RISK CATEGORY DISTRIBUTION ANALYSIS")
print("="*60)

print("\n1. Original Risk Categories:")
print(data['risk_category'].value_counts().sort_index())

print("\n2. Percentage Distribution:")
print(data['risk_category'].value_counts(normalize=True).sort_index() * 100)

print("\n3. Risk Score Statistics:")
print(data['composite_risk_score'].describe())

print("\n4. Risk Score Ranges:")
for category in data['risk_category'].unique():
    scores = data[data['risk_category'] == category]['composite_risk_score']
    print(f"{category:15s}: {scores.min():.2f} - {scores.max():.2f} (mean: {scores.mean():.2f})")
