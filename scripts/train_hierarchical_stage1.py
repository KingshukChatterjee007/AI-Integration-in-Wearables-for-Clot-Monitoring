import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
import os
from sklearn.metrics import classification_report, accuracy_score
from core.clot_hybrid_5class import ClotHybrid5Class
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, 
            weight=self.weight,
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def load_data_binary():
    logger.info("Loading Stage 1 (Binary Risk Gate) dataset...")
    train_df = pd.read_csv("processed_data/augmented_5class_train_gen.csv")
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    
    # Stage 1: Safeish (0,1,2) -> 0 | Risk (3,4) -> 1
    y_train = np.where(train_df['target'].values <= 2, 0, 1)
    y_test = np.where(test_df['target'].values <= 2, 0, 1)
    
    # Drop non-feature columns
    drop_cols = ['target', 'split', 'subject_id']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).values
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    
    logger.info(f"Target distribution (Binary): {np.bincount(y_train)}")
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data_binary()
    n_features = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # 2. Weighted Random Sampler
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weights = weights[y_train]
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 3. Model (Hierarchical Stage 1: Binary n_classes=2)
    model = ClotHybrid5Class(n_features=n_features, n_classes=2, dropout=0.3).to(device)
    
    # 4. Focal Loss (Safety-First: 5x weight for Risk)
    focal_weights = torch.FloatTensor([1.0, 5.0]).to(device)
    criterion = FocalLoss(weight=focal_weights, gamma=2.5, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    epochs = 40
    best_acc = 0
    os.makedirs("trained_models", exist_ok=True)
    best_model_path = "trained_models/hierarchical_stage1_gate.pth"
    
    logger.info("\n--- Training Stage 1: Binary Risk Gate ---")
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
        preds = []
        targets = []
        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                pred = torch.argmax(out, dim=1)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())
                
        val_acc = accuracy_score(targets, preds)
        logger.info(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            best_preds = preds
            best_targets = targets
            
    logger.info(f"\n--- Stage 1 Training Complete ---")
    logger.info(f"Best Val Accuracy: {best_acc*100:.2f}%")
    logger.info(f"Saved to: {best_model_path}")
    print("\n--- Stage 1 Binary Classification Report ---")
    print(classification_report(best_targets, best_preds, target_names=['Safe-ish', 'Risk']))

if __name__ == "__main__":
    main()
