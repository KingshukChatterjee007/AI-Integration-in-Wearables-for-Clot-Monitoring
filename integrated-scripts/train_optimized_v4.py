import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from transformer_stress_integrated import ClotTransformer, load_integrated_data
from focal_loss import ClassWeightedFocalLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_optimized_model(data_path, output_model_path):
    print(f"Loading temporal dataset: {data_path}")
    df = pd.read_csv(data_path)
    
    # Select numeric features and handle encoding
    # Handle NaNs which cause loss to go to NaN. fillna(0) is a standard baseline for this dataset.
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)
    X = numeric_df.drop(columns=['target', 'window_id'], errors='ignore').values
    y = numeric_df['target'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Scale data (StandardScaler recommended for Transformers)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    n_features = X_train.shape[1]
    print(f"Initializing model with {n_features} features...")
    model = ClotTransformer(n_features=n_features, n_classes=5).to('cpu')
    
    # SWAT Implementation: Use ClassWeightedFocalLoss
    # Weights: w=3.0 for Critical (Class 4)
    class_weights = [1.0, 1.0, 1.0, 1.0, 3.0]
    criterion = ClassWeightedFocalLoss(gamma=2.0, weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Training Loop (Simplified for demo, user can increase epochs)
    epochs = 10
    best_val_loss = float('inf')
    
    print("Starting precision-first training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.numpy())
                all_true.extend(batch_y.numpy())
        
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_model_path)
            print(f"Saving best model to {output_model_path}")

    # Post-training identification of MI threshold is the next logical step 
    # to be run with precision_optimizer.py using the new pth.
    print("\nTraining complete. Next step: Run precision_optimizer.py with the new pth to find MI threshold.")

if __name__ == "__main__":
    data_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_v4_TEMPORAL.csv"
    output_model = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\trained_models\clot_transformer_V4_OPTIMIZED.pth"
    
    if Path(data_file).exists():
        train_optimized_model(data_file, output_model)
    else:
        print("Temporal features not found. Please run temporal_features.py first.")
