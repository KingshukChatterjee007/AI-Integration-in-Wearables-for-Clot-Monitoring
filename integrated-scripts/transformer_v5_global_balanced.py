import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path

# =============================================================================
# 1. DATASET & ARCHITECTURE (v5 GLOBAL)
# =============================================================================

class IntegratedDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.randn(n_features, d_token))
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

class ClotTransformer(nn.Module):
    def __init__(self, n_features=8, n_classes=5, d_token=96, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, n_classes)
        )
    def forward(self, x):
        tokens = self.tokenizer(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        x = self.transformer(x)
        return self.classifier(self.norm(x[:, 0, :]))

# =============================================================================
# 2. TRAINING LOOP
# =============================================================================

def train_v5_global():
    data_path = 'processed_data/integrated_features_balanced_v2.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run ctgan_balancing.py first.")
        return

    print("--- Phase v5: Global Spectrum Training ---")
    df = pd.read_csv(data_path)
    
    # 8-Feature optimized subset
    feature_cols = ['eda_skew', 'eda_max', 'eda_peaks_count', 'eda_kurt', 'eda_std', 'eda_min', 'eda_range', 'eda_mean']
    X = df[feature_cols].values
    y = df['target'].values
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    train_loader = DataLoader(IntegratedDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(IntegratedDataset(X_test, y_test), batch_size=128)
    
    model = ClotTransformer(n_features=len(feature_cols))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Dataset: {len(df)} samples evenly balanced across 5 classes.")
    
    epochs = 50
    best_f1 = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Eval
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                out = model(batch_x)
                preds.extend(out.argmax(1).tolist())
                truths.extend(batch_y.tolist())
        
        report = classification_report(truths, preds, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Macro-F1: {macro_f1:.4f}")
            
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), 'trained_models/clot_transformer_global_v5.pth')

    print(f"\nTraining Complete. Best Macro-F1: {best_f1:.4f}")
    
    # Final Report
    final_model = ClotTransformer()
    final_model.load_state_dict(torch.load('trained_models/clot_transformer_global_v5.pth'))
    final_model.eval()
    
    preds, truths = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out = final_model(batch_x)
            preds.extend(out.argmax(1).tolist())
            truths.extend(batch_y.tolist())
            
    print("\n--- Final Global Performance (v5) ---")
    print(classification_report(truths, preds, target_names=['Low', 'Low-Mod', 'Mod', 'High', 'Critical']))

if __name__ == "__main__":
    train_v5_global()
