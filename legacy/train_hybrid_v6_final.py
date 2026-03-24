import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from clot_hybrid_v6 import ClotHybridV6
from pathlib import Path
import numpy as np

# OPTION 1: Balanced Clinical Training (1.0 : 2.5 : 5.0)
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_final_model():
    data_dir = Path("processed_data/v6_honest")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    X_train = torch.load(data_dir / "X_train_v6.pt")
    y_train = torch.load(data_dir / "y_train_v6.pt")
    X_val = torch.load(data_dir / "X_test_v6.pt") # Validation on unseen subjects
    y_val = torch.load(data_dir / "y_test_v6.pt")
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    
    # Initialize Model
    model = ClotHybridV6(n_features=X_train.shape[2], n_classes=3).to(device)
    
    # Re-calibrated Focal Loss (Option 1)
    # SAFE: 1.0, WARNING: 2.5, EMERGENCY: 5.0
    weights = torch.tensor([1.0, 2.5, 5.0]).to(device)
    criterion = WeightedFocalLoss(alpha=weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    print("Starting Final Production Training (Option 1: Balanced Weights)...")
    for epoch in range(30):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        
    torch.save(model.state_dict(), "trained_models/clot_hybrid_v6_final_balanced.pth")
    print("Training Complete. Model saved to trained_models/clot_hybrid_v6_final_balanced.pth")

if __name__ == "__main__":
    train_final_model()
