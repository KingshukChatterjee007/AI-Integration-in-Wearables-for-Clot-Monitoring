import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from pathlib import Path
import sys

# Setup Path
sys.path.append("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/legacy")
from clot_hybrid_v6 import ClotHybridV6

def run_honest_audit():
    data_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/processed_data/v6_honest")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Test Data
    X_test = torch.load(data_dir / "X_test_v6.pt")
    y_test = torch.load(data_dir / "y_test_v6.pt")
    
    # Load Recovered V6.1 Model
    model = ClotHybridV6(n_features=X_test.shape[2], n_classes=3)
    model.load_state_dict(torch.load("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/trained_models/clot_hybrid_v6_1_recovered.pth", map_location=device))
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1).numpy()
        
    final_preds = []
    # RESTORED V14 CLINICAL GATE (Sensitivity focus)
    for p in probs:
        p_safe, p_warn, p_emerg = p
        if p_emerg >= 0.5:
            final_preds.append(2)
        elif p_warn >= 0.35: # Custom Clinical Gate
            final_preds.append(1)
        else:
            final_preds.append(np.argmax(p))
            
    final_preds = np.array(final_preds)
    y_test_np = y_test.numpy()
    
    acc = np.mean(final_preds == y_test_np) * 100
    print(f"\n✅ UPDATED HONEST AUDIT ACCURACY: {acc:.2f}%\n")
    
    class_names = ['SAFE', 'WARNING', 'EMERGENCY']
    report = classification_report(y_test_np, final_preds, target_names=class_names, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # Confusion Matrix (Standard and Normalized)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    
    cm = confusion_matrix(y_test_np, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax[0],
                xticklabels=class_names, yticklabels=class_names)
    ax[0].set_title(f"Confusion Matrix (Acc={acc:.2f}%)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    
    cm_norm = confusion_matrix(y_test_np, final_preds, normalize='true')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges', ax=ax[1],
                xticklabels=class_names, yticklabels=class_names)
    ax[1].set_title("Normalized Confusion Matrix (Recall focus)")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    
    plt.tight_layout()
    plots_dir = Path("c:/Users/91704/AI-Integration-in-Wearables-for-Clot-Monitoring/model_comparison_plots_CLEAN/visualizations")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "final_clinical_audit_matrix.png")
    print(f"Matrix saved to {plots_dir / 'final_clinical_audit_matrix.png'}")

    # ROC Curves (One-vs-Rest)
    plt.figure(figsize=(10, 8))
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_np == i, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
    plt.legend(loc="lower right")
    plt.savefig(plots_dir / "final_roc_curves.png")
    print(f"ROC curves saved to {plots_dir / 'final_roc_curves.png'}")

if __name__ == "__main__":
    run_honest_audit()
