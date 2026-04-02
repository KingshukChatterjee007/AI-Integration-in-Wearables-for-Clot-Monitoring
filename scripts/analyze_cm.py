import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.clot_hybrid_5class import ClotHybrid5Class
from core.gating_logic_5class import apply_clinical_gate

def analyze_counts():
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_test = test_df['target'].values
    drop_cols = ['target', 'split', 'subject_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    model = ClotHybrid5Class(n_features=X_test.shape[1], n_classes=5).to(device)
    model.load_state_dict(torch.load("trained_models/clot_5class_hybrid_gen.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        out = model(X_test_t)
        preds, _ = apply_clinical_gate(out)
        preds = preds.cpu().numpy()
        
    cm = confusion_matrix(y_test, preds)
    target_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Crit']
    
    print("Confusion Matrix Counts:")
    print("True \\ Pred | " + " | ".join(target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:10} | " + " | ".join([f"{val:7}" for val in row]))

if __name__ == "__main__":
    analyze_counts()
