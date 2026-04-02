import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from core.clot_hybrid_5class import ClotHybrid5Class
from core.gating_logic_5class import apply_clinical_gate

def evaluate_model():
    print("Loading test data...")
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_test = test_df['target'].values
    drop_cols = ['target', 'split', 'subject_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = ClotHybrid5Class(n_features=X_test.shape[1], n_classes=5).to(device)
    model_path = "trained_models/clot_5class_hybrid_gen.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}. Please wait for training to finish.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_probs = []
    
    print("Running Inference with Clinical Gate...")
    for xb, yb in test_loader:
        out = model(xb)
        preds, probs = apply_clinical_gate(out)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
    target_names = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    
    print("\n================== 5-CLASS EVALUATION REPORT ==================")
    report_str = classification_report(y_test, all_preds, target_names=target_names)
    print(report_str)
    
    report_dict = classification_report(y_test, all_preds, target_names=target_names, output_dict=True)
    
    # Save text report
    os.makedirs('reports', exist_ok=True)
    with open("reports/5class_gated_results.txt", "w") as f:
        f.write("================== 5-CLASS EVALUATION REPORT ==================\n")
        f.write(report_str)
        
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('True Target')
    plt.xlabel('Predicted Output Gate')
    plt.title('Clinical Confusion Matrix (Threshold=0.45)')
    plt.savefig('reports/5class_confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix back to reports/5class_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
