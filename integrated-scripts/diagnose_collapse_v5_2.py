import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
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

def run_diagnosis():
    data_dir = Path("processed_data/v5_tensors")
    base_model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    latent_head_path = Path("trained_models/clot_latent_head_best.pth")
    
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    # Load CNN
    full_v5 = ClotHybridV5(n_features=X.shape[2])
    full_v5.load_state_dict(torch.load(base_model_path, map_location='cpu'))
    cnn = full_v5.cnn
    cnn.eval()
    
    # Load Head
    head = LatentHybridHead()
    head.load_state_dict(torch.load(latent_head_path, map_location='cpu'))
    head.eval()
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    
    def analyze_v5_2(indices, name):
        X_split = X[indices]
        y_split = y[indices].numpy()
        
        with torch.no_grad():
            z = cnn(X_split.transpose(1, 2))
            logits = head(z)
            preds = torch.argmax(logits, dim=-1).numpy()
            
        print(f"\n--- {name} Split Diagnosis (Phase 5.2 SMOTE) ---")
        unique_p, counts_p = np.unique(preds, return_counts=True)
        pred_map = {val: count for val, count in zip(unique_p, counts_p)}
        for i in range(5):
            print(f"  {class_names[i]}: {pred_map.get(i, 0)}")

    analyze_v5_2(train_idx, "TRAIN")
    analyze_v5_2(test_idx, "VALIDATION")

if __name__ == "__main__":
    run_diagnosis()
