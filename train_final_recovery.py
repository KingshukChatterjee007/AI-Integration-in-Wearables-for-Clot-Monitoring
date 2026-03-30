import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import logging
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Correct path for imports
sys.path.append("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/legacy")
from clot_hybrid_v6 import ClotHybridV6

class StandardFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(StandardFocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            weights = self.alpha[targets]
            focal_loss = weights * focal_loss
            
        return focal_loss.mean()

def train_recovery():
    data_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/processed_data/v6_honest")
    
    # Load Data
    X_train = torch.load(data_dir / "X_train_v11_augmented.pt")
    y_train = torch.load(data_dir / "y_train_v11_augmented.pt")
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt")
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    model = ClotHybridV6(n_features=X_train.shape[2], n_classes=3)
    
    # EMERGENCY PRIORITY: [1.0, 5.0, 30.0] 
    # High weight on EMERGENCY (30x) to ensure NO misses.
    weights = torch.tensor([1.0, 5.0, 30.0])
    criterion = StandardFocalLoss(alpha=weights, gamma=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 100
    best_val_acc = 0
    patience = 20
    no_improve = 0
    
    logger.info("Starting Recovery Training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation with GATED logic (matching user's clinical goal)
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                probs = torch.softmax(out, dim=1).numpy()
                batch_preds = []
                for p in probs:
                    if p[2] >= 0.35: # Clinical Sensitivity Gate for EMERGENCY
                        batch_preds.append(2)
                    elif p[1] >= 0.35: # Sensitive Gate for WARNING
                        batch_preds.append(1)
                    else:
                        batch_preds.append(np.argmax(p))
                all_preds.extend(batch_preds)
                all_targets.extend(yb.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        val_acc = np.mean(all_preds == all_targets)
        
        # Calculate EMERGENCY RECALL for monitoring
        emerg_idx = np.where(all_targets == 2)[0]
        emerg_recall = np.mean(all_preds[emerg_idx] == 2) if len(emerg_idx) > 0 else 0
        
        # Calculate SAFE RECALL
        safe_idx = np.where(all_targets == 0)[0]
        safe_recall = np.mean(all_preds[safe_idx] == 0) if len(safe_idx) > 0 else 0

        logger.info(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Emerg Rec: {emerg_recall:.2f} | Safe Rec: {safe_recall:.2f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "trained_models/clot_hybrid_v6_1_recovered.pth")
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            logger.info("Early stopping triggered.")
            break
            
    logger.info(f"Training Complete. Best Val Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    import numpy as np
    train_recovery()
