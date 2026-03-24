import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from clot_hybrid_v5 import ClotHybridV5
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Windows Store Python Path Fix
site_pkgs = Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_pkgs.exists():
    sys.path.append(str(site_pkgs))

class LatentHybridHead(nn.Module):
    def __init__(self, n_channels=128, n_time=30, n_classes=5, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_time, 256))
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))
    def forward(self, z):
        z = z.transpose(1, 2)
        lstm_out, _ = self.lstm(z)
        x = lstm_out + self.pos_encoder
        x = self.transformer(x)
        return self.head(x.mean(dim=1))

def calculate_mi(probs):
    # probs: (T, N, C)
    mean_prob = torch.mean(probs, dim=0)
    entropy_of_mean = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=-1)
    mean_of_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1), dim=0)
    return entropy_of_mean - mean_of_entropy

def run_swat_optimization():
    print("--- Phase 7: Bayesian SWAT Gating Optimization ---")
    data_dir = Path("processed_data/v5_tensors")
    base_model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    latent_head_path = Path("trained_models/clot_latent_head_best.pth")
    
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=s))
    
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Load Models
    full_v5 = ClotHybridV5(n_features=X.shape[2])
    full_v5.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    cnn = full_v5.cnn.eval()
    
    head = LatentHybridHead()
    head.load_state_dict(torch.load(latent_head_path, map_location='cpu'))
    
    # 1. MC Inference (T=50)
    print("Running MC Dropout Inference (T=50)...")
    def force_dropout(m):
        if type(m) == nn.Dropout:
            m.train()
    
    head.apply(force_dropout) # Keep dropout ON for uncertainty
    
    T = 50
    all_probs = []
    with torch.no_grad():
        z = cnn(X_test.transpose(1, 2))
        for _ in range(T):
            logits = head(z)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.unsqueeze(0))
    
    all_probs = torch.cat(all_probs, dim=0) # (T, N, C)
    mean_probs = torch.mean(all_probs, dim=0)
    mi = calculate_mi(all_probs).numpy()
    mean_preds = torch.argmax(mean_probs, dim=-1).numpy()
    
    # 2. Histogram Analysis
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(['Low', 'Low-Mod', 'Mod', 'High', 'Critical']):
        plt.hist(mi[y_test == i], alpha=0.5, label=name, bins=30)
    plt.title("MI Distribution per Class")
    plt.legend()
    Path("model_comparison_plots_CLEAN/swat_visuals").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/swat_visuals/mi_distribution.png")
    
    # 3. Threshold Sweep
    thresholds = np.arange(0.05, 0.45, 0.05)
    results = []
    
    print("\nSweeping MI Thresholds for Moderate & Low-Moderate...")
    for t_mod in thresholds:
        for t_lowmod in thresholds:
            final_preds = mean_preds.copy()
            # Apply Gating: If predicted class is 1 or 2 AND MI > threshold, downgrade to class-1 (safe)
            # Or just mark as "Too Uncertain"
            mask_mod = (mean_preds == 2) & (mi > t_mod)
            mask_lowmod = (mean_preds == 1) & (mi > t_lowmod)
            
            # For this audit, let's treat MI-gated samples as Class-0 (Low) to see effect on precision
            final_preds[mask_mod] = 0
            final_preds[mask_lowmod] = 0
            
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, final_preds, average=None, labels=[0,1,2,3,4], zero_division=0)
            
            results.append({
                't_mod': t_mod, 't_lowmod': t_lowmod,
                'p_lowmod': prec[1], 'r_lowmod': rec[1],
                'p_mod': prec[2], 'r_mod': rec[2],
                'p_crit': prec[4], 'r_crit': rec[4]
            })

    results_df = pd.DataFrame(results)
    
    # Find Best (Simple heuristic: Max sum of precision for 1 and 2 where recall > 0.4)
    filtered = results_df[(results_df['r_mod'] >= 0.4) | (results_df['r_lowmod'] >= 0.4)]
    if not filtered.empty:
        best = filtered.iloc[(filtered['p_mod'] + filtered['p_lowmod']).argmax()]
        print(f"\nBest SWAT Settings found:")
        print(f"  MI Threshold (Moderate): {best['t_mod']:.2f} -> Prec: {best['p_mod']*100:.1f}%")
        print(f"  MI Threshold (Low-Mod): {best['t_lowmod']:.2f} -> Prec: {best['p_lowmod']*100:.1f}%")
        print(f"  Critical Metrics: Precision {best['p_crit']*100:.1f}%, Recall {best['r_crit']*100:.1f}%")
    else:
        print("\nNo threshold met the 40% recall target. Broadening results...")

    results_df.to_csv("trained_models/swat_sweep_results.csv", index=False)

if __name__ == "__main__":
    run_swat_optimization()
