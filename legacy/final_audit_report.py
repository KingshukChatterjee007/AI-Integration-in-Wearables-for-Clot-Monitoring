import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
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

def run_final_audit():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    # Models
    full_v5 = ClotHybridV5(n_features=X.shape[2])
    full_v5.load_state_dict(torch.load("trained_models/clot_hybrid_v5_best.pth", map_location='cpu'))
    cnn = full_v5.cnn.eval()
    
    head = LatentHybridHead()
    head.load_state_dict(torch.load("trained_models/clot_latent_head_best.pth", map_location='cpu'))
    
    def evaluate(indices, mi_gate=False):
        X_split = X[indices]
        y_split = y[indices]
        
        # MC Inference for MI
        T = 50
        def force_dropout(m):
            if type(m) == nn.Dropout: m.train()
        head.apply(force_dropout)
        
        all_probs = []
        with torch.no_grad():
            z = cnn(X_split.transpose(1, 2))
            for _ in range(T):
                all_probs.append(torch.softmax(head(z), dim=-1).unsqueeze(0))
        
        all_probs = torch.cat(all_probs, dim=0)
        mean_probs = torch.mean(all_probs, dim=0)
        mi = calculate_mi(all_probs)
        preds = torch.argmax(mean_probs, dim=-1).numpy()
        
        if mi_gate:
            # Apply best thresholds from Phase 7 (0.20)
            mask_mod = (preds == 2) & (mi > 0.20)
            mask_lowmod = (preds == 1) & (mi > 0.20)
            preds[mask_mod] = 0
            preds[mask_lowmod] = 0
            
        return y_split, preds

    y_train, p_train = evaluate(train_idx, mi_gate=False)
    y_test, p_test = evaluate(test_idx, mi_gate=True)
    
    print("\n=== FINAL THESIS AUDIT: HYBRID V5.2 (SMOTE + SWAT) ===")
    print(f"Overall Train Accuracy: {accuracy_score(y_train, p_train)*100:.2f}%")
    print(f"Overall Test Accuracy (Subject-Wise): {accuracy_score(y_test, p_test)*100:.2f}%")
    print(f"Final Training Epoch: 30")
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    report = classification_report(y_test, p_test, target_names=class_names, output_dict=True, zero_division=0)
    
    print("\nPer-Class Breakdown (Test Set):")
    data = []
    for cls in class_names:
        metrics = report[cls]
        data.append({
            'Risk Category': cls,
            'Precision': f"{metrics['precision']*100:.1f}%",
            'Recall': f"{metrics['recall']*100:.1f}%",
            'F1-Score': f"{metrics['f1-score']*100:.1f}%",
            'Support': int(metrics['support'])
        })
    df = pd.DataFrame(data)
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_final_audit()
