"""
Debug Analysis: Root Cause Investigation for Poor Model Performance
"""

import pandas as pd
import numpy as np

# Load the improved dataset and analyze distribution issues
data = pd.read_csv('processed_data/integrated_features_improved.csv')

print("=== DIAGNOSIS: ROOT CAUSE ANALYSIS ===")
print(f"Total samples: {len(data)}")
print(f"Risk score range: {data['composite_risk_score'].min():.3f} to {data['composite_risk_score'].max():.3f}")
print(f"Risk score mean: {data['composite_risk_score'].mean():.3f}")
print(f"Risk score std: {data['composite_risk_score'].std():.3f}")

# Check class distribution for classification
bins = [0, 2, 4, 6, 11]
labels = [0, 1, 2, 3]
y_class = pd.cut(data['composite_risk_score'], bins=bins, labels=labels, include_lowest=True)

print("\n=== CLASS DISTRIBUTION ISSUE ===")
class_dist = y_class.value_counts().sort_index()
print("Classification distribution:")
for cls, count in class_dist.items():
    pct = count/len(data)*100
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

# Check train/test split impact
print("\n=== SUBJECT-BASED SPLIT ANALYSIS ===")
subjects = data['subject_id'].unique()
print(f"Unique subjects: {len(subjects)}")

# Simulate the subject-based split like in your model
train_subjects = subjects[:int(0.7 * len(subjects))]  # 70% subjects for training
test_subjects = subjects[int(0.7 * len(subjects)):]   # 30% subjects for testing

train_data = data[data['subject_id'].isin(train_subjects)]
test_data = data[data['subject_id'].isin(test_subjects)]

print(f"Training subjects: {len(train_subjects)} - {train_subjects}")
print(f"Testing subjects: {len(test_subjects)} - {test_subjects}")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# Check risk distribution in train vs test
train_class_dist = pd.cut(train_data['composite_risk_score'], bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
test_class_dist = pd.cut(test_data['composite_risk_score'], bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()

print(f"\n=== TRAIN/TEST CLASS IMBALANCE ===")
print("Training set class distribution:")
for cls in [0,1,2,3]:
    count = train_class_dist.get(cls, 0)
    pct = count/len(train_data)*100 if len(train_data) > 0 else 0
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

print("Test set class distribution:")
for cls in [0,1,2,3]:
    count = test_class_dist.get(cls, 0)
    pct = count/len(test_data)*100 if len(test_data) > 0 else 0
    print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

# Check if high-risk subjects are concentrated
print(f"\n=== HIGH-RISK CONCENTRATION ANALYSIS ===")
high_risk_subjects = data[data['composite_risk_score'] > 3]['subject_id'].value_counts()
print(f"Subjects with high risk scores (>3): {len(high_risk_subjects)}")
if len(high_risk_subjects) > 0:
    print(high_risk_subjects)

# Check feature variance
print(f"\n=== FEATURE VARIANCE ANALYSIS ===")
exclude_cols = ['subject_id', 'activity', 'window_id', 'composite_risk_score', 'composite_risk_score_old', 'gender', 'risk_category']
feature_columns = [col for col in data.columns if col not in exclude_cols]
X = data[feature_columns]

# Check for zero-variance features
zero_var_features = []
for col in X.columns:
    if X[col].std() == 0:
        zero_var_features.append(col)

print(f"Zero variance features: {len(zero_var_features)}")
if zero_var_features:
    print(f"  {zero_var_features[:10]}...")  # Show first 10

# Check feature correlation with target
target_corr = X.corrwith(data['composite_risk_score']).abs().sort_values(ascending=False)
print(f"\nTop 10 features correlated with target:")
print(target_corr.head(10))

print(f"\n=== CONCLUSION ===")
if test_class_dist.get(2, 0) == 0 and test_class_dist.get(3, 0) == 0:
    print("❌ MAJOR ISSUE: Test set has NO high/critical risk samples!")
    print("   This causes models to never see high-risk examples during validation.")

if train_class_dist.get(0, 0) > len(train_data) * 0.8:
    print("❌ MAJOR ISSUE: Training set is >80% low-risk samples!")
    print("   Severe class imbalance makes model biased toward low-risk predictions.")

# Check if test set has very few high-risk samples
high_risk_test = test_class_dist.get(2, 0) + test_class_dist.get(3, 0)
if high_risk_test < 20:
    print(f"❌ CRITICAL ISSUE: Test set has only {high_risk_test} high/critical risk samples!")
    print("   Insufficient high-risk samples for meaningful evaluation.")

print(f"\n🔧 RECOMMENDED FIXES:")
print("1. Use stratified train/test split instead of subject-based")
print("2. Apply SMOTE or other oversampling techniques")
print("3. Use class weights in models")
print("4. Consider ensemble methods")
print("5. Use appropriate metrics: F1-score, precision/recall, AUC")