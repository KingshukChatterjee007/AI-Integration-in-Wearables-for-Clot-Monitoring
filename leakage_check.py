import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data_path = Path(r'c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_enhanced_CLEAN.csv')
df = pd.read_csv(data_path)

non_feature_cols = ['subject_id', 'activity', 'window_id', 'risk_category']
X = df.drop(columns=non_feature_cols).select_dtypes(include=[np.number])
y = df['risk_category']

le = LabelEncoder()
y_enc = le.fit_transform(y)

# 1. Correlation with target (if any feature is > 0.9, it's likely leakage)
# Convert categorical target to numeric (dummy) to check correlation
target_numeric = y_enc
X_corr = X.copy()
X_corr['target'] = target_numeric
correlations = X_corr.corr()['target'].sort_values(ascending=False)

print("TOP CORRELATIONS WITH TARGET:")
print(correlations.head(20))
print("\nBOTTOM CORRELATIONS WITH TARGET:")
print(correlations.tail(20))

# 2. Random Forest Importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X.fillna(X.median()), y_enc)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nTOP 20 FEATURE IMPORTANCES (Random Forest):")
print(importances.head(20))

# 3. Check for obvious leakage names
leaky_keywords = ['risk', 'clot', 'category', 'label', 'target', 'score']
potential_leaky = [c for c in X.columns if any(k in c.lower() for k in leaky_keywords)]
print("\nPOTENTIALLY LEAKY NAMES DETECTED:")
print(potential_leaky)
