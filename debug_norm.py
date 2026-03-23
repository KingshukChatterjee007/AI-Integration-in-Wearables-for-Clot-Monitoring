import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def debug_data():
    file_path = 'processed_data/integrated_features_balanced_v1.csv'
    df = pd.read_csv(file_path)
    print(f"Total Rows: {len(df)}")
    
    df['subject_id'] = df['subject_id'].fillna('synthetic')
    subjects = df['subject_id'].astype(str).values
    
    non_features = ['subject_id', 'activity', 'window_id', 'risk_category', 'session', 'timestamp_start', 'target']
    feature_cols = [c for c in df.columns if c not in non_features]
    X_vals = df[feature_cols].values
    
    unique_subs = np.unique(subjects)
    print(f"Unique Subjects ({len(unique_subs)}): {unique_subs}")
    
    for sub in unique_subs:
        mask = (subjects == sub)
        count = mask.sum()
        slice_data = X_vals[mask]
        
        if count == 0:
            print(f"!!! CRITICAL: Subject '{sub}' has 0 samples according to mask!")
            continue
            
        print(f"Subject '{sub}': {count} samples, Slice shape: {slice_data.shape}")
        
        try:
            scaler = StandardScaler()
            scaler.fit_transform(slice_data)
        except Exception as e:
            print(f"!!! FAILED for Subject '{sub}': {e}")
            print(f"Slice Content (first 2 rows):\n{slice_data[:2]}")

if __name__ == "__main__":
    debug_data()
