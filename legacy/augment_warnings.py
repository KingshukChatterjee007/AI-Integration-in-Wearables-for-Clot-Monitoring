import torch
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def time_warp(x, sigma=0.2, knots=4):
    """
    Randomly stretches/compresses the sequence using cubic splines.
    """
    T, F = x.shape
    curr_t = np.linspace(0, 1, T)
    random_warp = np.cumsum(np.random.normal(loc=1.0, scale=sigma, size=knots))
    warp_steps = np.linspace(0, 1, knots)
    warp_fn = interp1d(warp_steps, random_warp, kind='cubic')(curr_t)
    
    # Scale back to T points
    time_pts = np.cumsum(warp_fn)
    time_pts = (time_pts - time_pts[0]) / (time_pts[-1] - time_pts[0]) * (T - 1)
    
    x_new = np.zeros_like(x)
    for f in range(F):
        x_new[:, f] = interp1d(np.arange(T), x[:, f], kind='linear')(time_pts)
    return x_new

def execute_warning_revival():
    data_dir = Path("processed_data/v6_honest")
    X_train = torch.load(data_dir / "X_train_v6.pt")
    y_train = torch.load(data_dir / "y_train_v6.pt")
    
    # Extract original WARNING (Class 1) samples (N=163)
    idx_w = (y_train == 1)
    X_w = X_train[idx_w].numpy()
    
    print(f"Original WARNING Count: {X_w.shape[0]}")
    
    augmented_X = []
    augmented_y = []
    
    target_count = 1000
    n_original = X_w.shape[0]
    repeats = (target_count // n_original) + 1
    
    for _ in range(repeats):
        for i in range(n_original):
            if len(augmented_X) >= target_count: break
            
            # Application: Time-Warping + Gaussian Noise
            warped = time_warp(X_w[i], sigma=0.05)
            jittered = warped + np.random.normal(0, 0.02, warped.shape)
            
            augmented_X.append(jittered)
            augmented_y.append(1)
            
    X_w_final = torch.tensor(np.array(augmented_X)).float()
    y_w_final = torch.tensor(np.array(augmented_y)).long()
    
    # Combine with others (SAFE=2000, EMERGENCY=423)
    # We already have augmented SAFE from previous step, we'll just merge
    X_safe = torch.load(data_dir / "X_train_v6_augmented.pt")
    y_safe = torch.load(data_dir / "y_train_v6_augmented.pt")
    
    X_s = X_safe[y_safe == 0]
    y_s = y_safe[y_safe == 0]
    
    X_e = X_train[y_train == 2]
    y_e = y_train[y_train == 2]
    
    X_final = torch.cat([X_s, X_w_final, X_e], dim=0)
    y_final = torch.cat([y_s, y_w_final, y_e], dim=0)
    
    torch.save(X_final, data_dir / "X_train_v11_augmented.pt")
    torch.save(y_final, data_dir / "y_train_v11_augmented.pt")
    
    print(f"Phase 11 Augmented Tensors ready: {X_final.shape}")
    print(f"Counts: SAFE (2000), WARNING (1000), EMERGENCY (423)")

if __name__ == "__main__":
    execute_warning_revival()
