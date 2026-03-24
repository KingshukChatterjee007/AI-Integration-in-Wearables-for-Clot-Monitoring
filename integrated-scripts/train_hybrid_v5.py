import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from pathlib import Path
import sys
import os

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clot_hybrid_v5 import ClotHybridV5

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        foc_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return foc_loss.mean()
        return foc_loss.sum()
from sklearn.model_selection import GroupShuffleSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_phase5():
    logger.info(f"Starting Phase 5.1 Hybrid Training (Subject-Wise Split) on {DEVICE}")
    
    # 1. Load Tensors
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    n_samples, seq_len, n_features = X.shape
    n_classes = 5
    
    # 2. Subject-Wise Split (GroupShuffleSplit)
    # This ensures no leakage: subjects in test are NEVER seen in train.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=s))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    logger.info(f"Train on {len(np.unique(s[train_idx]))} subjects, Test on {len(np.unique(s[test_idx]))} subjects")
    
    # 3. Class Balancing (WeightedRandomSampler)
    counts = np.bincount(y_train.numpy())
    weights = 1.0 / torch.FloatTensor(counts)
    sample_weights = weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    # 4. Initialize Model, Loss, Optimizer
    model = ClotHybridV5(n_features=n_features, n_classes=n_classes, seq_length=seq_len).to(DEVICE)
    # Increased Weight Decay (1e-1) for stronger regularization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    criterion = FocalLoss(gamma=2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Training Loop
    epochs = 40
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                pred = out.argmax(1)
                total += by.size(0)
                correct += (pred == by).sum().item()
                
        val_acc = 100 * correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            Path("trained_models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "trained_models/clot_hybrid_v5_best.pth")
            
        if (epoch+1) % 5 == 0:
            logger.info(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)")

    logger.info(f"Training Complete. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_phase5()
