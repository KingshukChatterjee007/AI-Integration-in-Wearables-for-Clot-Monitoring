import torch
import pandas as pd
import numpy as np
from transformer_stress_integrated import ClotTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def evaluate_optimized_model(model_path, data_path):
    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)
    X = numeric_df.drop(columns=['target', 'window_id'], errors='ignore').values
    y = numeric_df['target'].values
    
    # Same split as training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_val = scaler.transform(X_val)
    
    n_features = X_val.shape[1]
    model = ClotTransformer(n_features=n_features, n_classes=5).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val))
        preds = torch.argmax(logits, dim=-1).numpy()
        
    print("=== PHASE 4 (SWAT-Tier) EVALUATION ===")
    print(classification_report(y_val, preds, target_names=['Low', 'Low-Mod', 'Mod', 'High', 'Critical']))
    
    # Global metrics
    acc = accuracy_score(y_val, preds)
    print(f"Global Accuracy: {acc:.4f}")

if __name__ == "__main__":
    data_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_v4_TEMPORAL.csv"
    model_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\trained_models\clot_transformer_V4_OPTIMIZED.pth"
    evaluate_optimized_model(model_file, data_file)
