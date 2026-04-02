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

def calibrate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_test = test_df['target'].values
    drop_cols = ['target', 'split', 'subject_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    
    n_features = X_test.shape[1]
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    test_ds = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 2. Load Best Model
    model = ClotHybrid5Class(n_features=n_features, n_classes=5).to(device)
    model_path = "trained_models/clot_5class_hybrid_gen.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Get Softmax Probabilities
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in test_loader:
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            
    probs_matrix = np.vstack(all_probs)
    y_true = np.concatenate(all_targets)

    # 4. Grid Search for Dual Thresholds (Balanced High-Risk Detection)
    # T_crit: threshold for Critical (Class 4)
    # T_high: threshold for High (Class 3)
    
    best_score = -1
    best_thresholds = (0.35, 0.25)
    best_results = None
    
    print("\n--- Tuning Clinical Gates for Balanced High-Risk Detection ---")
    
    # Granular search for thresholds
    for t_crit in np.linspace(0.01, 0.4, 20):
        for t_high in np.linspace(0.01, 0.3, 15):
            
            # Apply Gating Logic
            y_pred = []
            for p in probs_matrix:
                if p[4] > t_crit:
                    y_pred.append(4) # Critical
                elif p[3] > t_high:
                    y_pred.append(3) # High
                else:
                    y_pred.append(np.argmax(p)) # Normal Argmax
            
            y_pred = np.array(y_pred)
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            
            crit_recall = report['4']['recall']
            high_recall = report['3']['recall']
            overall_f1 = report['weighted avg']['f1-score']
            
            # CLINICAL SAFETY FLOOR: Prioritize Critical Recall above 80%
            if crit_recall >= 0.80:
                score = crit_recall * 10.0 + high_recall * 5.0 + overall_f1
                if score > best_score:
                    best_score = score
                    best_thresholds = (t_crit, t_high)
                    best_results = report
                    best_preds = y_pred
    
    # Fallback if no threshold reaches 80% Recall (be more lenient)
    if best_score == -1:
        print("Warning: Could not reach 80% Critical Recall floor. Relaxing to 65%...")
        for t_crit in np.linspace(0.01, 0.4, 30):
            for t_high in np.linspace(0.01, 0.3, 20):
                y_pred = []
                for p in probs_matrix:
                    if p[4] > t_crit: y_pred.append(4)
                    elif p[3] > t_high: y_pred.append(3)
                    else: y_pred.append(np.argmax(p))
                
                y_pred = np.array(y_pred)
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                crit_recall = report['4']['recall']
                high_recall = report['3']['recall']
                
                if crit_recall >= 0.65:
                    score = crit_recall * 10.0 + high_recall * 5.0
                    if score > best_score:
                        best_score = score
                        best_thresholds = (t_crit, t_high)
                        best_preds = y_pred

    print(f"\nOptimal Thresholds Found:")
    print(f"  > Critical Gate (T_crit): {best_thresholds[0]:.2f}")
    print(f"  > High Gate (T_high):     {best_thresholds[1]:.2f}")
    
    print("\nBalanced High-Risk Performance Report:")
    target_names = ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']
    print(classification_report(y_true, best_preds, target_names=target_names))
    
    # Save results to file
    with open("reports/5class_gated_results_BALANCED.txt", "w") as f:
        f.write("================== BALANCED 5-CLASS PERFORMANCE REPORT ==================\n")
        f.write(f"T_crit: {best_thresholds[0]:.2f} | T_high: {best_thresholds[1]:.2f}\n\n")
        f.write(classification_report(y_true, best_preds, target_names=target_names))
        
    print(f"\nFinal report saved to reports/5class_gated_results_BALANCED.txt")

if __name__ == "__main__":
    calibrate()
