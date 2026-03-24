import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from clot_hybrid_v6 import ClotHybridV6
from pathlib import Path
import sys

# Windows Store Python Path Fix
site_pkgs = Path.home() / "AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages"
if site_pkgs.exists():
    sys.path.append(str(site_pkgs))

def run_v6_audit():
    data_dir = Path("processed_data/v6_honest")
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt").numpy()
    
    model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
    model.load_state_dict(torch.load("trained_models/clot_hybrid_v6_augmented.pth", map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        out = model(X_test)
        preds = torch.argmax(out, dim=1).numpy()
    
    print("\n=== PHASE 9 FINAL AUDIT: 'HONEST' 3-CLASS SYSTEM ===")
    print(f"Overall Accuracy (Unseen Subjects 9 & 10): {accuracy_score(y_test, preds)*100:.2f}%")
    
    class_names = ['SAFE', 'WARNING', 'EMERGENCY']
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=class_names))
    
    # 3x3 Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title("PHASE 9: Normalized 3x3 Confusion Matrix (Honest Tiers)")
    plt.ylabel("Actual Tier")
    plt.xlabel("Predicted Tier")
    
    Path("model_comparison_plots_CLEAN/visualizations").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/v6_honest_confusion_matrix.png")
    print("\nHeatmap saved to model_comparison_plots_CLEAN/visualizations/v6_honest_confusion_matrix.png")
    
    print("\nNumerical 3x3 Matrix (Raw Counts):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names))

if __name__ == "__main__":
    run_v6_audit()
