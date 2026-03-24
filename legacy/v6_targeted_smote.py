import torch
import numpy as np
from pathlib import Path
import sys
# Windows Store Python Path Fix
site_pkgs = Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_pkgs.exists():
    if str(site_pkgs) not in sys.path:
        sys.path.append(str(site_pkgs))

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print(f"Error: imbalanced-learn not found in {site_pkgs}")
    sys.exit(1)

def augment_honest_tiers():
    data_dir = Path("processed_data/v6_honest")
    X_train = torch.load(data_dir / "X_train_v6.pt")
    y_train = torch.load(data_dir / "y_train_v6.pt")
    
    # Flatten for SMOTE: (N, 30, 188) -> (N, 5640)
    N, T, F = X_train.shape
    X_flat = X_train.view(N, -1).numpy()
    y_np = y_train.numpy()
    
    print(f"Original Train Counts: {np.unique(y_np, return_counts=True)}")
    
    # Target Counts: SAFE (0) -> 2000, WARNING (1) -> 1000, EMERGENCY (2) -> 423 (unchanged)
    # Note: SMOTE cannot target above existing max unless we use specific strategy
    # If original max is 423, we can only go to 2000 if we set sampling_strategy
    target_dict = {0: 2000, 1: 1000, 2: 423}
    
    smote = SMOTE(sampling_strategy=target_dict, random_state=42)
    X_res, y_res = smote.fit_resample(X_flat, y_np)
    
    # Reshape back: (N_new, 5640) -> (N_new, 30, 188)
    X_res_tensor = torch.tensor(X_res).view(-1, T, F).float()
    y_res_tensor = torch.tensor(y_res).long()
    
    print(f"Augmented Train Counts: {np.unique(y_res, return_counts=True)}")
    
    # Save Augmented Set
    save_path = data_dir / "X_train_v6_augmented.pt"
    torch.save(X_res_tensor, save_path)
    torch.save(y_res_tensor, data_dir / "y_train_v6_augmented.pt")
    
    print(f"Augmented V6 Tensors saved to {data_dir}")

if __name__ == "__main__":
    augment_honest_tiers()
