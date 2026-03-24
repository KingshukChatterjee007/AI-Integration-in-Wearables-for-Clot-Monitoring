import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging
from transformer_stress_integrated import ClotTransformer, load_integrated_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def enable_dropout_only(model):
    """
    STRICT IMPLEMENTATION RULE: Enable ONLY nn.Dropout while keeping 
    LayerNorm/BatchNorm in eval mode to prevent statistics corruption.
    """
    model.eval() # Start with eval mode
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train() # Force dropout to remain active

def calculate_uncertainty(probs_list):
    """
    MATHEMATICAL IMPLEMENTATION:
    T = num_samples
    Predictive Mean: p_hat = 1/T * sum(p_t)
    Predictive Entropy: H = -sum(p_hat * log(p_hat))
    Mutual Information: MI = H - 1/T * sum(H_individual)
    """
    # probs_list shape: (T, Batch, Classes)
    T = probs_list.shape[0]
    
    # 1. Predictive Mean Class Probabilities
    p_hat = torch.mean(probs_list, dim=0) # (Batch, Classes)
    
    # 2. Predictive Entropy (Total Uncertainty)
    # Adding epsilon to avoid log(0)
    eps = 1e-10
    entropy = -torch.sum(p_hat * torch.log(p_hat + eps), dim=1)
    
    # 3. Expected Entropy (Aleatoric Uncertainty)
    individual_entropies = -torch.sum(probs_list * torch.log(probs_list + eps), dim=2)
    expected_entropy = torch.mean(individual_entropies, dim=0)
    
    # 4. Mutual Information (Epistemic Uncertainty - Lack of Knowledge)
    mutual_info = entropy - expected_entropy
    
    return p_hat, entropy, mutual_info

@torch.no_grad()
def mc_dropout_inference(model, x_tensor, T=50, mi_threshold=0.4):
    """
    Full MC Dropout Inference Pipeline with Clinical Routing Gate.
    """
    enable_dropout_only(model)
    
    all_probs = []
    for _ in range(T):
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        
    all_probs = torch.stack(all_probs) # (T, Batch, Classes)
    
    p_hat, entropy, mi = calculate_uncertainty(all_probs)
    
    # Final Predictions
    pred_classes = torch.argmax(p_hat, dim=1)
    
    results = []
    for i in range(x_tensor.size(0)):
        class_idx = pred_classes[i].item()
        confidence = p_hat[i, class_idx].item()
        mi_val = mi[i].item()
        
        # CLINICAL ROUTING GATE
        if mi_val > mi_threshold:
            status = f"UNCERTAIN - LOW CONFIDENCE (MI={mi_val:.4f})"
            final_call = "FLAG FOR CLINICAL REVIEW"
        else:
            status = "CONFIDENT"
            final_call = f"Risk Level: {class_idx}" # Placeholder for class name mapping
            
        results.append({
            'prediction': class_idx,
            'confidence': confidence,
            'entropy': entropy[i].item(),
            'mutual_info': mi_val,
            'status': status,
            'final_call': final_call
        })
        
    return results

def run_mc_evaluation():
    logger.info("Starting Bayesian MC-Dropout Evaluation Pipeline")
    
    # 1. Load Data and Model
    X, y, subjects, class_names, feature_names = load_integrated_data()
    n_features = X.shape[1]
    
    model = ClotTransformer(n_features=n_features, n_classes=len(class_names)).to(DEVICE)
    model_path = Path('trained_models/clot_transformer_integrated_best.pth')
    
    if not model_path.exists():
        logger.error("No trained model found. Please train v3 model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # 2. Select samples for analysis (Critical and Uncertain cases)
    # We take 100 samples from the test set
    indices = np.random.choice(len(X), 100, replace=False)
    X_sample = torch.FloatTensor(X[indices]).to(DEVICE)
    y_true = y[indices]
    
    # 3. Perform Stochastic Inference
    logger.info(f"Running T=50 Stochastic Forward Passes for {len(indices)} samples...")
    batch_results = mc_dropout_inference(model, X_sample, T=50)
    
    # 4. Display Results with Clinical Gate
    header = f"{'True':<12} | {'Pred':<12} | {'Conf':<6} | {'MI':<7} | {'Status/Call'}"
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    
    for i, res in enumerate(batch_results):
        true_label = class_names[y_true[i]]
        pred_label = class_names[res['prediction']]
        
        status_str = f"{res['status']} -> {res['final_call']}"
        logger.info(f"{true_label:<12} | {pred_label:<12} | {res['confidence']:.2f} | {res['mutual_info']:.4f} | {status_str}")

if __name__ == "__main__":
    run_mc_evaluation()
