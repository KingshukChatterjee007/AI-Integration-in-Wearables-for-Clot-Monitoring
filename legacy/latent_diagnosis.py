import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from clot_hybrid_v5 import ClotHybridV5
from pathlib import Path

def analyze_latent_space():
    data_dir = Path("processed_data/v5_tensors")
    model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups=s))
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    
    model = ClotHybridV5(n_features=X.shape[2])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        # X: (N, Time, Features) -> (N, Features, Time)
        X_in = X_train.transpose(1, 2)
        # Extract features from CNN (MultiScaleCNN)
        z = model.cnn(X_in) # (N, Channels, Compressed_Time)
        z_flat = z.view(z.size(0), -1).numpy()
        
    print(f"Latent feature shape: {z_flat.shape}")
    
    # Check for mean distance between classes
    unique_y = np.unique(y_train)
    means = []
    for cls in unique_y:
        means.append(z_flat[y_train == cls].mean(axis=0))
    
    # Distance between Low (0) and Moderate (2)
    dist_0_2 = np.linalg.norm(means[0] - means[2])
    # Distance between Mod (2) and Critical (4)
    dist_2_4 = np.linalg.norm(means[2] - means[4])
    
    print(f"Distance (Low vs Mod): {dist_0_2:.4f}")
    print(f"Distance (Mod vs Critical): {dist_2_4:.4f}")
    
    if dist_0_2 < 0.1 * dist_2_4:
        print("CRITICAL: Latent space is collapsed for Low/Moderate. Simple SMOTE will fail.")
        return False
    else:
        print("Latent space shows class separation. SMOTE is viable.")
        return True

if __name__ == "__main__":
    analyze_latent_space()
