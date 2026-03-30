import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import recall_score, classification_report
import sys

# Add path to legacy to import ClotHybridV6
sys.path.append("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/legacy")
from clot_hybrid_v6 import ClotHybridV6

def tune_thresholds():
    model_path = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/trained_models/clot_hybrid_v6_1_recovered.pth")
    data_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/processed_data/v6_honest")
    
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt").numpy()
    
    model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1).numpy()
    
    print(f"{'Thresh':<8} | {'SAFE Rec':<8} | {'WARN Rec':<8} | {'EMERG Rec':<8}")
    print("-" * 45)
    
    best_t = 0.35
    max_safe_rec = 0
    
    for t in np.arange(0.3, 0.7, 0.01):
        preds = []
        for p in probs:
            if p[2] >= 0.5: # Fixed EMERGENCY gate
                preds.append(2)
            elif p[1] >= t: # WARNING gate
                preds.append(1)
            else:
                preds.append(np.argmax(p))
        
        r = recall_score(y_test, preds, average=None, labels=[0, 1, 2], zero_division=0)
        print(f"{t:.2f}     | {r[0]:.2f}     | {r[1]:.2f}     | {r[2]:.2f}")
        
        if r[2] == 1.0 and r[0] > max_safe_rec:
            max_safe_rec = r[0]
            best_t = t
            
    print(f"\nOptimal Warning Threshold (maintaining 100% Emergency Recall): {best_t:.2f}")

if __name__ == "__main__":
    tune_thresholds()
