import torch
import numpy as np
import pandas as pd
from pathlib import Path

def create_honest_tiers():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt").numpy()
    
    # 1. 3-Class Mapping
    # Original: 0=Low, 1=Low-Mod, 2=Mod, 3=High, 4=Crit
    # New: 0=SAFE (0,1), 1=WARNING (2,3), 2=EMERGENCY (4)
    y_honest = torch.zeros_like(y)
    y_honest[y == 1] = 0 # SAFE
    y_honest[y == 2] = 1 # WARNING
    y_honest[y == 3] = 1 # WARNING
    y_honest[y == 4] = 2 # EMERGENCY
    
    # 2. Strict Subject-Wise Split (Zero Leakage)
    # Subjects are 1-indexed (1 to 10 in the dataset usually)
    # We will use Subjects 9 and 10 for Testing. Subjects 1-8 for Training.
    test_subjects = [9.0, 10.0]
    
    test_mask = np.isin(s, test_subjects)
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y_honest[train_mask]
    X_test, y_test = X[test_mask], y_honest[test_mask]
    s_train, s_test = s[train_mask], s[test_mask]
    
    print(f"Honest Split Complete:")
    print(f"  Training: {X_train.shape} (Subjects: {np.unique(s_train)})")
    print(f"  Testing:  {X_test.shape} (Subjects: {np.unique(s_test)})")
    
    # Save the Honest dataset
    save_dir = Path("processed_data/v6_honest")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, save_dir / "X_train_v6.pt")
    torch.save(y_train, save_dir / "y_train_v6.pt")
    torch.save(X_test, save_dir / "X_test_v6.pt")
    torch.save(y_test, save_dir / "y_test_v6.pt")
    print(f"V6 Tensors saved to {save_dir}")
    
    # Simple class distribution audit
    classes, counts = torch.unique(y_train, return_counts=True)
    print("Train Class Distribution (SAFE, WARNING, EMERGENCY):", counts.tolist())

if __name__ == "__main__":
    create_honest_tiers()
