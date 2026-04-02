import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from core.clot_hybrid_5class import ClotHybrid5Class

def run_diagnostic():
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
        probs = torch.softmax(out, dim=1).cpu().numpy()
        
    crit_probs = probs[:, 4]
    high_probs = probs[:, 3]
    
    # Class 3 (High) True samples
    mask_high = (y_test == 3)
    # Class 4 (Crit) True samples
    mask_crit = (y_test == 4)
    
    print("--- HIGH TRUTH ANALYSIS ---")
    print(f"Mean P(High) for True High: {high_probs[mask_high].mean():.4f}")
    print(f"Mean P(Crit) for True High: {crit_probs[mask_high].mean():.4f}")
    print(f"Max P(High) for True High: {high_probs[mask_high].max():.4f}")
    
    print("\n--- CRITICAL TRUTH ANALYSIS ---")
    print(f"Mean P(Crit) for True Crit: {crit_probs[mask_crit].mean():.4f}")
    print(f"Mean P(High) for True Crit: {high_probs[mask_crit].mean():.4f}")
    print(f"Max P(Crit) for True Crit: {crit_probs[mask_crit].max():.4f}")
    
    print("\n--- GLOBAL THRESHOLD DIAGNOSTIC ---")
    print(f"Samples with P(High) > 0.30: {(high_probs > 0.30).sum()}")
    print(f"Samples with P(Crit) > 0.20: {(crit_probs > 0.20).sum()}")

if __name__ == "__main__":
    run_diagnostic()
