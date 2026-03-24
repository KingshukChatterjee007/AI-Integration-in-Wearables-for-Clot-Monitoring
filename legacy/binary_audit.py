import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
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

def run_binary_audit():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=s))
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Models
    cnn = ClotHybridV5(n_features=X.shape[2]).cnn
    cnn.load_state_dict(torch.load("trained_models/clot_hybrid_v5_best.pth", map_location='cpu'), strict=False)
    cnn.eval()
    
    head = LatentHybridHead()
    head.load_state_dict(torch.load("trained_models/clot_latent_head_best.pth", map_location='cpu'))
    head.eval()
    
    with torch.no_grad():
        z = cnn(X_test.transpose(1, 2))
        logits = head(z)
        preds = torch.argmax(logits, dim=-1).numpy()

    # BINARY MAPPING:
    # Non-Critical (0, 1, 2) -> 0
    # Critical (3, 4) -> 1
    y_bin = np.where(y_test <= 2, 0, 1)
    p_bin = np.where(preds <= 2, 0, 1)
    
    print("\n=== BINARY THESIS AUDIT: NON-CRITICAL vs CRITICAL ===")
    print(f"Binary Test Accuracy: {accuracy_score(y_bin, p_bin)*100:.2f}%")
    
    target_names = ['Non-Critical (Stable)', 'Critical (Clot Risk)']
    print(classification_report(y_bin, p_bin, target_names=target_names))
    
    cm = confusion_matrix(y_bin, p_bin)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    run_binary_audit()
