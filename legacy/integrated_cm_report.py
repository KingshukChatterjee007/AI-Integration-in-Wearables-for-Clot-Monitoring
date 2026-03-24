import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from clot_hybrid_v5 import ClotHybridV5
from pathlib import Path
import sys

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
    mean_prob = torch.mean(probs, dim=0)
    entropy_of_mean = -torch.sum(mean_prob * torch.log(mean_prob + 1e-10), dim=-1)
    mean_of_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1), dim=0)
    return (entropy_of_mean - mean_of_entropy).numpy()

def run_cm_audit():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=s))
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Models
    full_v5 = ClotHybridV5(n_features=X.shape[2])
    full_v5.load_state_dict(torch.load("trained_models/clot_hybrid_v5_best.pth", map_location='cpu'))
    cnn = full_v5.cnn.eval()
    
    head = LatentHybridHead()
    head.load_state_dict(torch.load("trained_models/clot_latent_head_best.pth", map_location='cpu'))
    head.eval()
    
    # MC Inference for MI (T=50)
    T = 50
    def force_dropout(m):
        if type(m) == nn.Dropout: m.train()
    head.apply(force_dropout)
    
    all_probs = []
    with torch.no_grad():
        z = cnn(X_test.transpose(1, 2))
        for _ in range(T):
            all_probs.append(torch.softmax(head(z), dim=-1).unsqueeze(0))
    
    all_probs = torch.cat(all_probs, dim=0)
    mean_probs = torch.mean(all_probs, dim=0)
    mi = calculate_mi(all_probs)
    preds = torch.argmax(mean_probs, dim=-1).numpy()
    
    # Binary Gating logic (from Phase 7 audit)
    mask_mod = (preds == 2) & (mi > 0.20)
    mask_lowmod = (preds == 1) & (mi > 0.20)
    preds[mask_mod] = 0
    preds[mask_lowmod] = 0
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2, 3, 4])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix: Integrated (SMOTE + SWAT)')
    
    Path("model_comparison_plots_CLEAN/visualizations").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/integrated_confusion_matrix.png")
    
    print("\n--- Numerical Confusion Matrix (Critical Class Focus) ---")
    print(f"Total Critical Samples in Test set: {cm[4].sum()}")
    print(f"Correctly Predicted Critical (TP): {cm[4, 4]}")
    print(f"Critical Predicted as High (Adjacent): {cm[4, 3]}")
    print(f"Critical Predicted as Moderate: {cm[4, 2]}")
    print(f"Critical Missed (FN): {cm[4].sum() - cm[4, 4]}")
    
    print("\nFull Normalized Matrix:")
    print(cm_norm)

if __name__ == "__main__":
    run_cm_audit()
