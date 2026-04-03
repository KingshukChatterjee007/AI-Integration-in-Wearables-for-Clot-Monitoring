import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import os

def rehabilitate_baseline_manual_calib():
    print("--- Phase 1: Data Preparation ---")
    DATA_PATH = 'processed_data/integrated_features_enhanced_CLEAN.csv'
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Map Risk Category to Integers
    risk_map = {'Low':0, 'Low-Moderate':1, 'Moderate':2, 'High':3, 'Critical':4}
    df['target'] = df['risk_category'].map(risk_map)
    
    # Stratified Split (70 Train, 15 Val, 15 Test)
    drop_cols = ['target', 'risk_category', 'split', 'subject_id', 'activity', 'window_id', 'data_source']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['target']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)

    # SMOTE Balancing
    print("\n--- Phase 2: SMOTE Class Balancing ---")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # XGBoost Training
    print("\n--- Phase 3: Baseline XGBoost Training ---")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softprob', num_class=5, random_state=42)
    xgb.fit(X_train_sm, y_train_sm)

    # Phase 4: Manual Isotonic Calibration (using Validation Set)
    print("\n--- Phase 4: Manual Isotonic Calibration (Validation Set) ---")
    val_probs_raw = xgb.predict_proba(X_val)
    test_probs_raw = xgb.predict_proba(X_test)
    
    # One Isotonic Regression per class
    calibrators = []
    val_probs_calib = np.zeros_like(val_probs_raw)
    test_probs_calib = np.zeros_like(test_probs_raw)
    
    for i in range(5):
        # Target for class i is binary (is it class i or not?)
        y_val_bin = (y_val == i).astype(int)
        iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
        # We fit on the raw probability of being class i
        iso.fit(val_probs_raw[:, i], y_val_bin)
        calibrators.append(iso)
        
        # Apply to Val and Test
        val_probs_calib[:, i] = iso.transform(val_probs_raw[:, i])
        test_probs_calib[:, i] = iso.transform(test_probs_raw[:, i])
        
    # Re-normalize calibrated probabilities to sum to 1
    val_probs_calib = val_probs_calib / val_probs_calib.sum(axis=1)[:, None]
    test_probs_calib = test_probs_calib / test_probs_calib.sum(axis=1)[:, None]

    # Threshold Optimization on VAL set (Find T_crit for Recall >= 0.95)
    print("\n--- Phase 5: Precision-Recall Threshold Sweep (VAL) ---")
    best_t_crit = 0.5
    best_p_at_r95 = -1
    
    for t_crit in np.linspace(0.001, 0.9, 200):
        y_val_pred = []
        for p in val_probs_calib:
            if p[4] > t_crit: y_val_pred.append(4)
            else: y_val_pred.append(np.argmax(p))
        
        rep = classification_report(y_val, y_val_pred, labels=[0,1,2,3,4], output_dict=True, zero_division=0)
        rec_crit = rep['4']['recall']
        prec_crit = rep['4']['precision']
        
        if rec_crit >= 0.95:
            if prec_crit > best_p_at_r95:
                # We want 95% recall + maximum precision
                best_p_at_r95 = prec_crit
                best_t_crit = t_crit

    print(f"Optimal Critical Threshold found: {best_t_crit:.4f} (Val Precision: {best_p_at_r95:.2%})")

    # Final Evaluation (TEST SET)
    print("\n--- Phase 6: Final Evaluation (TEST) ---")
    y_test_pred = []
    for p in test_probs_calib:
        if p[4] > best_t_crit: y_test_pred.append(4)
        else: y_test_pred.append(np.argmax(p))
            
    test_report = classification_report(y_test, y_test_pred, target_names=['Low', 'Low-Mod', 'Mod', 'High', 'Critical'])
    print(test_report)
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nFinal Confusion Matrix (Test):")
    print(cm)

    # Save Results
    os.makedirs("reports", exist_ok=True)
    with open("reports/REHABILITATED_BASELINE_REPORT.txt", "w") as f:
        f.write("================== REHABILITATED BASELINE PERFORMANCE REPORT ==================\n")
        f.write(f"Technique: XGBoost + SMOTE + Manual Multi-Class Isotonic Calibration\n")
        f.write(f"Objective: Critical Recall >= 0.95 on Validation\n")
        f.write(f"Optimal Threshold (Critical): {best_t_crit:.4f}\n\n")
        f.write(test_report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

if __name__ == "__main__":
    rehabilitate_baseline_manual_calib()
