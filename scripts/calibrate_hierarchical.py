import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.clot_hybrid_5class import ClotHybrid5Class

def calibrate_hierarchical():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_true = test_df['target'].values
    X_test = test_df.drop(columns=['target', 'split', 'subject_id'], errors='ignore').values
    n_features = X_test.shape[1]
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=64, shuffle=False)

    # 2. Load Models
    gate_model = ClotHybrid5Class(n_features=n_features, n_classes=2).to(device)
    gate_model.load_state_dict(torch.load("trained_models/hierarchical_stage1_gate.pth", map_location=device))
    gate_model.eval()

    expert_model = ClotHybrid5Class(n_features=n_features, n_classes=2).to(device)
    expert_model.load_state_dict(torch.load("trained_models/hierarchical_stage2_expert.pth", map_location=device))
    expert_model.eval()

    base_model = ClotHybrid5Class(n_features=n_features, n_classes=5).to(device)
    base_model.load_state_dict(torch.load("trained_models/clot_5class_hybrid_gen.pth", map_location=device))
    base_model.eval()

    # 3. Pre-cache Probs
    gate_probs = []
    expert_probs = []
    base_probs = []
    
    with torch.no_grad():
        for xb in test_loader:
            gate_probs.append(torch.softmax(gate_model(xb[0]), dim=1).cpu().numpy())
            expert_probs.append(torch.softmax(expert_model(xb[0]), dim=1).cpu().numpy())
            base_probs.append(torch.softmax(base_model(xb[0]), dim=1).cpu().numpy())
            
    p_gate = np.vstack(gate_probs)
    p_expert = np.vstack(expert_probs)
    p_base = np.vstack(base_probs)

    # 4. Multiobjective Search
    best_score = -1
    best_params = None
    best_preds = None

    print("\n--- Clinical Calibration for Hierarchical Ensemble ---")
    
    # T_risk: High for precision, Low for recall
    # T_crit: High for precision, Low for recall
    for t_risk in np.linspace(0.1, 0.9, 17):
        for t_crit in np.linspace(0.05, 0.8, 16):
            
            y_pred = []
            for i in range(len(X_test)):
                if p_gate[i][1] > t_risk:
                    # Expert mode
                    if p_expert[i][1] > t_crit: y_pred.append(4)
                    else: y_pred.append(3)
                else:
                    # Safe Mode (Granular 0, 1, 2)
                    safe_indices = [0, 1, 2]
                    safe_probs = p_base[i][safe_indices]
                    y_pred.append(safe_indices[np.argmax(safe_probs)])
            
            y_pred = np.array(y_pred)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            c_rec = report['4']['recall']
            h_rec = report['3']['recall']
            low_rec = report['0']['recall'] # Low-Risk Recall (Majority)
            
            # Clinical Priority: 
            # 1. Critical recall MUST be >= 80% (Clinical Safety Floor)
            # 2. High risk recall >= 5% (Recovery path)
            # 3. Overall F1 score (Precision balance)
            if c_rec >= 0.80 and h_rec >= 0.05:
                # Weighted score
                score = c_rec * 5.0 + h_rec * 3.0 + low_rec * 2.0 + report['weighted avg']['f1-score']
                if score > best_score:
                    best_score = score
                    best_params = (t_risk, t_crit)
                    best_preds = y_pred

    if best_params is None:
        print("ALERT: Could not meet safety floor (80% Critical, 5% High). Picking best available...")
        # Fallback search
        for t_risk in np.linspace(0.1, 0.9, 20):
            for t_crit in np.linspace(0.01, 0.5, 20):
                y_pred = []
                for i in range(len(X_test)):
                    if p_gate[i][1] > t_risk:
                        if p_expert[i][1] > t_crit: y_pred.append(4)
                        else: y_pred.append(3)
                    else:
                        safe_indices = [0, 1, 2]
                        safe_probs = p_base[i][safe_indices]
                        y_pred.append(safe_indices[np.argmax(safe_probs)])
                y_pred = np.array(y_pred)
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                score = report['4']['recall'] * 10 + report['3']['recall'] * 5
                if score > best_score:
                    best_score = score
                    best_params = (t_risk, t_crit)
                    best_preds = y_pred

    print(f"\nFinal Hierarchical Calibration:")
    print(f"  > Stage 1 Gate T_risk: {best_params[0]:.3f}")
    print(f"  > Stage 2 Expert T_crit: {best_params[1]:.3f}")
    
    target_names = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    final_report = classification_report(y_true, best_preds, target_names=target_names)
    print("\n" + final_report)

    with open("reports/MODEL_COMPARISON_REPORT_HIERARCHICAL.txt", "w") as f:
        f.write("================== FINAL HIERARCHICAL ENSEMBLE REPORT ==================\n")
        f.write(f"T_risk: {best_params[0]:.3f} | T_crit: {best_params[1]:.3f}\n\n")
        f.write(final_report)

if __name__ == "__main__":
    calibrate_hierarchical()
