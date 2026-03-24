import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from clot_hybrid_v5 import ClotHybridV5
from pathlib import Path

def run_diagnosis():
    data_dir = Path("processed_data/v5_tensors")
    model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    model = ClotHybridV5(n_features=X.shape[2])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    
    def analyze_split(indices, name):
        X_split = X[indices]
        y_split = y[indices].numpy()
        
        with torch.no_grad():
            logits = model(X_split)
            preds = torch.argmax(logits, dim=-1).numpy()
            
        print(f"\n--- {name} Split Diagnosis ---")
        print(f"Total Samples: {len(indices)}")
        print(f"Total Subjects: {len(np.unique(s[indices]))}")
        
        print("\nTrue Class Distribution (Ground Truth):")
        unique_y, counts_y = np.unique(y_split, return_counts=True)
        for val, count in zip(unique_y, counts_y):
            print(f"  {class_names[val]}: {count}")
            
        print("\nPredicted Class Distribution:")
        unique_p, counts_p = np.unique(preds, return_counts=True)
        pred_map = {val: count for val, count in zip(unique_p, counts_p)}
        for i in range(5):
            print(f"  {class_names[i]}: {pred_map.get(i, 0)}")

    analyze_split(train_idx, "TRAIN")
    analyze_split(test_idx, "VALIDATION")

if __name__ == "__main__":
    run_diagnosis()
