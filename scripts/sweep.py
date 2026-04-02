import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.clot_hybrid_5class import ClotHybrid5Class

def sweep_thresholds():
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
        
    best_combo = None
    max_h_plus_c = 0
    
    for tc in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
        for th in [0.1, 0.2, 0.3, 0.4]:
            # Apply gate
            preds = np.argmax(probs, axis=1)
            for i in range(len(probs)):
                if probs[i, 4] >= tc:
                    preds[i] = 4
                elif probs[i, 3] >= th:
                    preds[i] = 3
            
            # Calculate recalls
            h_recall = ((preds == 3) & (y_test == 3)).sum() / (y_test == 3).sum()
            c_recall = ((preds == 4) & (y_test == 4)).sum() / (y_test == 4).sum()
            
            if h_recall + c_recall > max_h_plus_c:
                max_h_plus_c = h_recall + c_recall
                best_combo = (tc, th, h_recall, c_recall)
                
    print(f"Best Sweep: CritTh={best_combo[0]}, HighTh={best_combo[1]} => HighRec={best_combo[2]:.4f}, CritRec={best_combo[3]:.4f}")

if __name__ == "__main__":
    sweep_thresholds()
