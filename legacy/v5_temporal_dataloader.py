import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_temporal_dataset(csv_path, seq_length=30, overlap=15, feature_set='all'):
    """
    Creates 3D temporal tensors (N, Seq, Features) from a CSV dataset.
    Groups data by subject_id before windowing to prevent leakage across subjects.
    """
    logger.info(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort to ensure temporal order within subjects
    if 'window_id' in df.columns:
        df = df.sort_values(by=['subject_id', 'window_id']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['subject_id']).reset_index(drop=True)
        
    non_features = ['subject_id', 'activity', 'window_id', 'risk_category', 'session', 'timestamp_start', 'target', 'Label', 'subject']
    
    if feature_set == 'optimized_8':
        feature_cols = ['eda_skew', 'eda_max', 'eda_peaks_count', 'eda_kurt', 'eda_std', 'eda_min', 'eda_range', 'eda_mean']
    else:
        # Automatically select numeric columns and exclude non_features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in non_features]
        
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    X_list = []
    y_list = []
    s_list = [] # Subject IDs
    
    subjects = df['subject_id'].unique()
    # Map subject strings to integers for PyTorch tensor compatibility
    sub_map = {sub: i for i, sub in enumerate(subjects)}
    step = seq_length - overlap
    
    for sub in subjects:
        sub_df = df[df['subject_id'] == sub].copy()
        
        # Normalize per subject (Crucial for wearable sensors)
        scaler = StandardScaler()
        sub_vals = scaler.fit_transform(sub_df[feature_cols].values)
        sub_targets = sub_df['target'].values
        
        # Sliding window
        for i in range(0, len(sub_vals) - seq_length + 1, step):
            X_window = sub_vals[i : i + seq_length]
            y_label = sub_targets[i + seq_length - 1]
            
            X_list.append(X_window)
            y_list.append(y_label)
            s_list.append(sub_map[sub])
            
    X = np.array(X_list)
    y = np.array(y_list)
    s = np.array(s_list)
    
    logger.info(f"Generated {X.shape[0]} sequences from {len(subjects)} subjects of shape {X.shape[1:]}")
    return X, y, s, feature_cols, sub_map

if __name__ == "__main__":
    data_path = Path(r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_balanced_v2.csv")
    if data_path.exists():
        X, y, s, features, sub_map = create_temporal_dataset(data_path)
        
        # Save tensors for training
        output_dir = Path("processed_data/v5_tensors")
        output_dir.mkdir(exist_ok=True)
        
        torch.save(torch.FloatTensor(X), output_dir / "X_v5_30seq.pt")
        torch.save(torch.LongTensor(y), output_dir / "y_v5_30seq.pt")
        torch.save(torch.LongTensor(s), output_dir / "subjects_v5_30seq.pt")
        
        # Save feature names for reference in training script
        with open(output_dir / "feature_names.txt", "w") as f:
            f.write("\n".join(features))
            
        logger.info(f"Saved tensors to {output_dir}")
    else:
        logger.error(f"Path not found: {data_path}")
