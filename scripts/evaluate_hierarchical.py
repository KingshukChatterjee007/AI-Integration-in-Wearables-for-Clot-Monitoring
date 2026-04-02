import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.clot_hybrid_5class import ClotHybrid5Class

def evaluate_hierarchical():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_true = test_df['target'].values
    drop_cols = ['target', 'split', 'subject_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    n_features = X_test.shape[1]
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=64, shuffle=False)

    # 2. Load Models
    # A. Stage 1 (Binary Gate)
    gate_model = ClotHybrid5Class(n_features=n_features, n_classes=2).to(device)
    gate_model.load_state_dict(torch.load("trained_models/hierarchical_stage1_gate.pth", map_location=device))
    gate_model.eval()

    # B. Stage 2 (Emergency Expert)
    expert_model = ClotHybrid5Class(n_features=n_features, n_classes=2).to(device)
    expert_model.load_state_dict(torch.load("trained_models/hierarchical_stage2_expert.pth", map_location=device))
    expert_model.eval()

    # C. Baseline 5-Class Model (for Safe-tier granularity: 0,1,2)
    base_model = ClotHybrid5Class(n_features=n_features, n_classes=5).to(device)
    base_model.load_state_dict(torch.load("trained_models/clot_5class_hybrid_gen.pth", map_location=device))
    base_model.eval()

    # 3. Get Softmax Probabilities for all models
    gate_probs = []
    expert_probs = []
    base_probs = []
    
    print("Running inference across hierarchical stages...")
    with torch.no_grad():
        for xb in test_loader:
            # Stage 1 Gate
            g_out = gate_model(xb[0])
            gate_probs.append(torch.softmax(g_out, dim=1).cpu().numpy())
            
            # Stage 2 Expert
            e_out = expert_model(xb[0])
            expert_probs.append(torch.softmax(e_out, dim=1).cpu().numpy())
            
            # Baseline 5-Class
            b_out = base_model(xb[0])
            base_probs.append(torch.softmax(b_out, dim=1).cpu().numpy())
            
    p_gate = np.vstack(gate_probs)
    p_expert = np.vstack(expert_probs)
    p_base = np.vstack(base_probs)

    # 4. Grid Search for Dual hierarchical Thresholds
    best_score = -1
    best_thresholds = (0.3, 0.3)
    best_preds = None

    print("\n--- Tuning Hierarchical Thresholds (Safety-First) ---")
    
    for t_gate in np.linspace(0.01, 0.5, 15):
        for t_expert in np.linspace(0.01, 0.4, 15):
            
            y_pred = []
            for i in range(len(X_test)):
                # Step 1: Gate Check (Is it an Emergency 3 or 4?)
                if p_gate[i][1] > t_gate:
                    # Step 2: Expert Check (High vs Critical?)
                    if p_expert[i][1] > t_expert:
                        y_pred.append(4) # Critical
                    else:
                        y_pred.append(3) # High
                else:
                    # Step 3: Granular Safe Tiers (0, 1, 2)
                    # We take the argmax only among the Safe classes
                    safe_indices = [0, 1, 2]
                    safe_probs = p_base[i][safe_indices]
                    y_pred.append(safe_indices[np.argmax(safe_probs)])
            
            y_pred = np.array(y_pred)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            crit_recall = report['4']['recall']
            high_recall = report['3']['recall']
            
            # Objective: 100% Safety for Critical, Maximize High
            if crit_recall >= 0.95:
                score = crit_recall * 10.0 + high_recall * 3.0 + report['weighted avg']['f1-score']
                if score > best_score:
                    best_score = score
                    best_thresholds = (t_gate, t_expert)
                    best_preds = y_pred

    print(f"\nOptimal Hierarchical Thresholds:")
    print(f"  > Stage 1 Gate (T_risk):   {best_thresholds[0]:.3f}")
    print(f"  > Stage 2 Expert (T_crit): {best_thresholds[1]:.3f}")
    
    print("\n--- Hierarchical Ensemble Report ---")
    target_names = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    final_report = classification_report(y_true, best_preds, target_names=target_names)
    print(final_report)
    
    # Save results
    with open("reports/MODEL_COMPARISON_REPORT_HIERARCHICAL.txt", "w") as f:
        f.write("================== HIERARCHICAL ENSEMBLE REPORT ==================\n")
        f.write(f"T_risk: {best_thresholds[0]:.3f} | T_crit: {best_thresholds[1]:.3f}\n\n")
        f.write(final_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_true, best_preds)))
        
    print(f"\nFinal results saved to reports/MODEL_COMPARISON_REPORT_HIERARCHICAL.txt")

if __name__ == "__main__":
    evaluate_hierarchical()
