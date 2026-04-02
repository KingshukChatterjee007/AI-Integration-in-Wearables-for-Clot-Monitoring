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

def load_data():
    logger.info("Loading augmented 5-class dataset...")
    train_df = pd.read_csv("processed_data/augmented_5class_train_gen.csv")
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    
    # Drop non-feature columns
    drop_cols = ['target', 'split', 'subject_id']
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).values
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    
    logger.info(f"Train samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    logger.info(f"Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = load_data()
    n_features = X_train.shape[1]
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # 2. Weighted Random Sampler (20% balance per batch)
    class_counts = np.bincount(y_train)
    weights = 1. / class_counts
    samples_weights = weights[y_train]
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 3. Model
    model = ClotHybrid5Class(n_features=n_features, n_classes=5, dropout=0.3).to(device)
    
    # 4. Focal Loss & Optimizer
    focal_weights = torch.FloatTensor([1.0, 2.0, 3.0, 10.0, 20.0]).to(device)
    criterion = FocalLoss(weight=focal_weights, gamma=2.5, label_smoothing=0.08) # Increased Gamma and Smoothing
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    epochs = 50
    best_acc = 0
    os.makedirs("trained_models", exist_ok=True)
    best_model_path = "trained_models/clot_5class_hybrid_gen.pth"
    
    logger.info("\n--- Starting Generative Training ---")
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
            
    logger.info(f"\n--- Training Complete ---")
    logger.info(f"Best Val Accuracy: {best_acc*100:.2f}%")
    logger.info(f"Saved to: {best_model_path}")
    
    print("\n--- Final Classification Report ---")
    target_names = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    print(classification_report(best_targets, best_preds, target_names=target_names))

if __name__ == "__main__":
    main()
