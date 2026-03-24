import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from clot_hybrid_v6 import ClotHybridV6
from pathlib import Path

def run_thresholded_audit():
    data_dir = Path("processed_data/v6_honest")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Test Data
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt")
    
    # Load Model v6.1 Recovery
    model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
    model.load_state_dict(torch.load("trained_models/clot_hybrid_v6_1_recovered.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        probs = F.softmax(logits, dim=1).numpy()
        
    y_test_np = y_test.numpy()
    final_preds = []
    
    # Apply Threshold Logic: SAFE=0, WARNING=1, EMERGENCY=2
    for p in probs:
        p_safe, p_warn, p_emerg = p
        
        # Priority 1: Emergency (Standard argmax if high)
        if p_emerg >= 0.5:
            final_preds.append(2)
        # Priority 2: Warning Gating (Force warning if > 0.35)
        elif p_warn >= 0.35:
            final_preds.append(1)
        # Priority 3: Default to argmax
        else:
            final_preds.append(np.argmax(p))
            
    final_preds = np.array(final_preds)
    
    # Generate Report
    report = classification_report(y_test_np, final_preds, target_names=['SAFE', 'WARNING', 'EMERGENCY'], zero_division=0)
    cm = confusion_matrix(y_test_np, final_preds)
    
    output_text = f"=== PHASE 11: WARNING RECOVERY AUDIT (Threshold=0.35) ===\n\n"
    output_text += report + "\n\nConfusion Matrix:\n" + str(cm)
    
    with open("model_comparison_plots_CLEAN/PHASE_11_THRESHOLD_REPORT.txt", "w") as f:
        f.write(output_text)
    
    print(output_text)
    
    # VISUALIZATION
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm / cm.sum(axis=1)[:, None], annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=['SAFE', 'WARNING', 'EMERGENCY'],
                yticklabels=['SAFE', 'WARNING', 'EMERGENCY'])
    plt.title("PHASE 11: Final 3x3 Diagonal Matrix (Warning Revived)")
    plt.xlabel("Predicted Clinical Tier")
    plt.ylabel("Actual Clinical Tier")
    plt.tight_layout()
    plt.savefig("model_comparison_plots_CLEAN/visualizations/v11_threshold_matrix.png")
    plt.close()

if __name__ == "__main__":
    run_thresholded_audit()
