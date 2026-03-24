import torch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from clot_hybrid_v5 import ClotHybridV5
from pathlib import Path
import sys

# Windows Store Python Path Fix
site_pkgs = Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_pkgs.exists():
    sys.path.append(str(site_pkgs))

def calculate_cosine_similarity():
    data_dir = Path("processed_data/v5_tensors")
    model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y, groups=s))
    
    X_train, y_train = X[train_idx], y[train_idx]
    
    model = ClotHybridV5(n_features=X.shape[2])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        z = model.cnn(X_train.transpose(1, 2))
        z_flat = z.view(z.size(0), -1) # (N, D)
        
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    centroids = []
    
    for i in range(5):
        mask = (y_train == i)
        if mask.any():
            centroid = z_flat[mask].mean(dim=0)
            centroids.append(centroid)
        else:
            centroids.append(torch.zeros(z_flat.size(1)))
            
    centroids = torch.stack(centroids)
    
    # Calculate Cosine Similarity Matrix
    # S(A, B) = (A . B) / (||A|| * ||B||)
    cos_sim = F.cosine_similarity(centroids.unsqueeze(1), centroids.unsqueeze(0), dim=-1)
    
    print("\n=== Latent Space Cosine Similarity Matrix ===")
    print("           " + "".join([f"{name:>10}" for name in class_names]))
    for i, name in enumerate(class_names):
        row = "".join([f"{cos_sim[i, j]:>10.4f}" for j in range(5)])
        print(f"{name:>10}: {row}")

if __name__ == "__main__":
    calculate_cosine_similarity()
