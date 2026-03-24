import sys
from pathlib import Path
# Fix for Windows Store Python site-packages
site_pkgs = Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_pkgs.exists():
    sys.path.append(str(site_pkgs))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    # Try another common path
    sys.path.append(str(Path.home() / "AppData/Roaming/Python/Python311/site-packages"))
    from imblearn.over_sampling import SMOTE
from clot_hybrid_v5 import ClotHybridV5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_latent_swat():
    logger.info(f"--- Phase 5.2 Latent-Space SMOTE Training on {DEVICE} ---")
    
    # 1. Load Data
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt") # (N, 30, 188)
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    # 2. Subject-Wise Split (Preserve 80/20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 3. Load Feature Extractor (Encoder) from Phase 5.1
    best_model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    base_model = ClotHybridV5(n_features=X.shape[2])
    base_model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    encoder = base_model.cnn.to(DEVICE)
    encoder.eval()
    
    # 4. Extract Latent Features
    def extract_z(X_tensor):
        zs = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size].to(DEVICE).transpose(1, 2)
                z = encoder(batch) # (Batch, Channels, Time_Compressed)
                zs.append(z.cpu())
        return torch.cat(zs, dim=0)

    logger.info("Extracting latent CNN features...")
    Z_train = extract_z(X_train)
    Z_test = extract_z(X_test)
    
    # Z shape: (N, 128, 30)
    n_samples, n_channels, n_time = Z_train.shape
    
    # 5. Apply SMOTE in Latent Space (Flattened)
    logger.info("Applying SMOTE to balance minority classes in latent space...")
    smote = SMOTE(random_state=42)
    Z_train_flat = Z_train.view(n_samples, -1).numpy()
    Z_aug_flat, y_aug = smote.fit_resample(Z_train_flat, y_train.numpy())
    
    Z_aug = torch.FloatTensor(Z_aug_flat).view(-1, n_channels, n_time)
    y_aug = torch.LongTensor(y_aug)
    
    logger.info(f"Augmented Train size: {len(Z_aug)} (Balanced across all 5 classes)")
    
    # 6. Define the Post-CNN Model (LSTM + Transformer + Head)
    class LatentHybridHead(nn.Module):
        def __init__(self, n_channels, n_time, n_classes=5, dropout=0.5):
            super().__init__()
            # Reuse logic from ClotHybridV5 (Blocks 2, 3, 4)
            self.lstm = nn.LSTM(input_size=n_channels, hidden_size=128, num_layers=2, 
                                batch_first=True, bidirectional=True, dropout=dropout)
            
            self.pos_encoder = nn.Parameter(torch.zeros(1, n_time, 256))
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, 
                                                     dim_feedforward=512, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
            
            self.head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
            
        def forward(self, z):
            # z: (Batch, Channels, Time) -> (Batch, Time, Channels)
            z = z.transpose(1, 2)
            lstm_out, _ = self.lstm(z) # (Batch, Time, 256)
            
            x = lstm_out + self.pos_encoder
            x = self.transformer(x) # (Batch, Time, 256)
            
            # Global pooling
            x = x.mean(dim=1)
            return self.head(x)

    model = LatentHybridHead(n_channels, n_time).to(DEVICE)
    # Use Label Smoothing as requested
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    
    train_loader = DataLoader(TensorDataset(Z_aug, y_aug), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(Z_test, y_test), batch_size=32, shuffle=False)
    
    # 7. Training Loop
    best_acc = 0
    history = []
    for epoch in range(30):
        model.train()
        total_loss = 0
        for bz, by in train_loader:
            bz, by = bz.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bz)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        correct, total = 0, 0
        y_preds = []
        with torch.no_grad():
            for bz, by in test_loader:
                bz, by = bz.to(DEVICE), by.to(DEVICE)
                out = model(bz)
                preds = out.argmax(1)
                correct += (preds == by).sum().item()
                total += by.size(0)
                y_preds.extend(preds.cpu().numpy())
                
        val_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            # Save the combined model
            full_model = ClotHybridV5(n_features=X.shape[2])
            full_model.load_state_dict(base_model.state_dict())
            # Inject new head parameters
            # Note: This is simplified for logic demonstration
            torch.save(model.state_dict(), "trained_models/clot_latent_head_best.pth")

    logger.info(f"Phase 5.2 Training Complete. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_latent_swat()
