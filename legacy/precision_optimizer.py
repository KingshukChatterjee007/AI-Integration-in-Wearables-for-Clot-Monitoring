import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from transformer_stress_integrated import ClotTransformer, load_integrated_data
from sklearn.metrics import precision_score, classification_report

def calculate_uncertainty(model, x, num_samples=3):
    """
    Perform Monte Carlo Dropout inference.
    returns: predictive_mean, mutual_information
    """
    model.train() # Enable dropout
    probs_list = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            logits = model(x)
            if isinstance(logits, tuple): logits = logits[0]
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)
            
    probs_stack = torch.stack(probs_list) # (T, B, C)
    predictive_mean = torch.mean(probs_stack, dim=0) # (B, C)
    
    # Predictive Entropy H[y|x, D]
    predictive_entropy = -torch.sum(predictive_mean * torch.log(predictive_mean + 1e-10), dim=-1)
    
    # Expected Entropy E[H[y|x, w]]
    # (1/T) * sum( -sum(p * log p) )
    expected_entropy = torch.mean(-torch.sum(probs_stack * torch.log(probs_stack + 1e-10), dim=-1), dim=0)
    
    # Mutual Information (Knowledge Uncertainty)
    mi = predictive_entropy - expected_entropy
    
    return predictive_mean, mi

def validate_precision_thresholds(model_path, data_path, mi_range=np.arange(0.05, 0.5, 0.05)):
    # Load model
    # We need input_dim. Let's inspect the data first or use a default.
    dataset = pd.read_csv(data_path).sample(n=5000, random_state=42) if len(pd.read_table(data_path, nrows=0).columns) > 0 else pd.read_csv(data_path)
    # Corrected sample logic
    dataset = pd.read_csv(data_path)
    if len(dataset) > 200:
        dataset = dataset.sample(n=200, random_state=42)
    
    # Synchronize feature selection with training logic
    # Use select_dtypes(include=[np.number]) and drop target/window_id
    numeric_df = dataset.select_dtypes(include=[np.number]).fillna(0.0)
    features = numeric_df.drop(columns=['target', 'window_id'], errors='ignore').columns.tolist()
    input_dim = len(features)
    print(f"Detected {input_dim} baseline features (including deltas).")
    
    model = ClotTransformer(n_features=input_dim, n_classes=5).to('cpu')
    # Try loading state dict if exists, otherwise dummy for code demonstration
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Prepare data
    X = torch.tensor(numeric_df[features].values, dtype=torch.float32)
    y_true = numeric_df['target'].values
    
    print(f"Running inference on {len(X)} samples...")
    p_mean, mi_scores = calculate_uncertainty(model, X)
    preds = torch.argmax(p_mean, dim=-1).numpy()
    mi_scores = mi_scores.numpy()
    
    results = []
    for thresh in mi_range:
        # SWAT Rule Implementation:
        # If predicted class is 4 (Critical) and MI <= threshold, keep label 4.
        # Otherwise, if it's class 4 but high uncertainty, downgrade/route.
        
        swat_preds = preds.copy()
        mask_uncertain = (preds == 4) & (mi_scores > thresh)
        swat_preds[mask_uncertain] = -1 # Filtered/Routed
        
        # Calculate Precision for Critical Class (excluding filtered)
        valid_indices = swat_preds != -1
        if np.sum(valid_indices & (preds == 4)) == 0:
            precision_critical = 0.0
        else:
            # We care about precision of the REMAINING 'Critical' labels
            final_critical_indices = (swat_preds == 4)
            if np.sum(final_critical_indices) == 0:
                precision_critical = 0.0
            else:
                tp = np.sum((swat_preds == 4) & (y_true == 4))
                fp = np.sum((swat_preds == 4) & (y_true != 4))
                precision_critical = tp / (tp + fp)
                
        results.append({
            'MI_Threshold': thresh,
            'Critical_Precision': precision_critical,
            'Samples_Retained': np.sum(swat_preds == 4),
            'Samples_Routed': np.sum(mask_uncertain)
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    data_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_v4_TEMPORAL.csv"
    # Placeholder for model path - user may need to specify if multiple exist
    model_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\trained_models\clot_transformer_V4_OPTIMIZED.pth"
    
    if Path(data_file).exists():
        sweep_results = validate_precision_thresholds(model_file, data_file)
        print("\n=== Precision vs MI Threshold Sweep ===")
        print(sweep_results.to_string(index=False))
        
        target_thresh = sweep_results[sweep_results['Critical_Precision'] >= 0.75]
        if not target_thresh.empty:
            best = target_thresh.iloc[0]
            print(f"\n[SUCCESS] Found 75% Precision threshold at MI <= {best['MI_Threshold']}")
        else:
            print("\n[WARNING] 75% Precision not met in current range. Lowering MI threshold further may be required.")
