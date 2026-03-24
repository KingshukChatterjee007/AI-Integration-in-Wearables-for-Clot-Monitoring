import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from clot_hybrid_v6 import ClotHybridV6
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha # Weights for each class
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train_v6():
    data_dir = Path("processed_data/v6_honest")
    X_train = torch.load(data_dir / "X_train_v6.pt")
    y_train = torch.load(data_dir / "y_train_v6.pt")
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt")
    
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    model = ClotHybridV6(n_features=X_train.shape[2], n_classes=3)
    
    # Class weights: EMERGENCY (2) = 3.0, WARNING (1) = 1.5, SAFE (0) = 1.0
    weights = torch.tensor([1.0, 1.5, 3.0])
    criterion = WeightedFocalLoss(alpha=weights, gamma=2.0)
    
    # Adam with weight_decay=0.1
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-1)
    
    epochs = 30
    best_acc = 0
    
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
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        
        val_acc = correct / total
        logger.info(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "trained_models/clot_hybrid_v6_honest.pth")
            
    logger.info(f"V6 Training Complete. Best Val Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_v6()
