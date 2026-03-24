import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from transformer_stress_integrated import ClotTransformer, load_integrated_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_integrated_gradients(model, input_tensor, target_class, baseline=None, steps=50):
    """
    Simplified Integrated Gradients to explain Transformer decisions.
    Calculates which features most influenced the 'target_class' prediction.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    
    # Path from baseline to input
    alphas = torch.linspace(0, 1, steps).to(DEVICE)
    integrated_grads = torch.zeros_like(input_tensor)
    
    for alpha in alphas:
        # Interpolate
        x_step = baseline + alpha * (input_tensor - baseline)
        x_step.requires_grad = True
        
        logits = model(x_step)
        score = logits[0, target_class]
        
        model.zero_grad()
        score.backward()
        integrated_grads += x_step.grad / steps
        
    # Attribute = (input - baseline) * integrated_grads
    attributions = (input_tensor - baseline) * integrated_grads
    return attributions.detach().cpu().numpy()

def main():
    logger.info("Starting Clot-Stress XAI Attribution Analysis")
    
    # 1. Load Data and Model
    X, y, subjects, class_names, feature_names = load_integrated_data()
    n_features = X.shape[1]
    
    model = ClotTransformer(n_features=n_features, n_classes=len(class_names)).to(DEVICE)
    model_path = Path('trained_models/clot_transformer_integrated_best.pth')
    
    if not model_path.exists():
        logger.error(f"Trained model not found at {model_path}. Run training first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 2. Find "Critical" and "High" Risk samples to explain
    results_dir = Path('model_comparison_plots_CLEAN/attention_maps')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    critical_idx = np.where(class_names == 'Critical')[0][0]
    high_idx = np.where(class_names == 'High')[0][0]
    
    # Analyze first 5 Critical cases and first 5 High cases
    test_indices = []
    for target in [critical_idx, high_idx]:
        indices = np.where(y == target)[0][:3]
        test_indices.extend(indices)
        
    logger.info(f"Generating explanations for {len(test_indices)} clinical events...")
    
    all_attributions = []
    
    for idx in test_indices:
        input_x = torch.FloatTensor(X[idx]).unsqueeze(0).to(DEVICE)
        target_y = y[idx]
        
        # Get Model Prediction
        with torch.no_grad():
            outputs = model(input_x)
            pred_idx = outputs.argmax(1).item()
            
        # Get Attributions (Explanation)
        attr = get_integrated_gradients(model, input_x, target_y)
        all_attributions.append(attr.flatten())
        
        # Plot Heatmap for this specific alert
        plt.figure(figsize=(12, 6))
        
        # Sort features by absolute impact
        sorted_indices = np.argsort(np.abs(attr.flatten()))[::-1]
        top_n = 10
        top_indices = sorted_indices[:top_n]
        
        top_features = [feature_names[i] for i in top_indices]
        top_scores = attr.flatten()[top_indices]
        
        sns.barplot(x=top_scores, y=top_features, palette='vlag')
        plt.title(f"Clinical Rationale: {class_names[target_y]} Alert (Sub: {subjects[idx]})\nPrediction: {class_names[pred_idx]}")
        plt.xlabel("Impact on Risk Score (Integrated Gradients)")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        fname = results_dir / f"explanation_{subjects[idx]}_{class_names[target_y]}_{idx}.png"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        
    # Summary Heatmap
    plt.figure(figsize=(14, 8))
    summary_df = pd.DataFrame(all_attributions, columns=feature_names)
    avg_impact = summary_df.abs().mean().sort_values(ascending=False).head(15)
    
    sns.barplot(x=avg_impact.values, y=avg_impact.index, palette='magma')
    plt.title("Top 15 Physiological Drivers of Integrated Clot Risk")
    plt.xlabel("Average Absolute Attribution (Importance)")
    
    summary_fname = results_dir / "global_feature_importance.png"
    plt.tight_layout()
    plt.savefig(summary_fname)
    plt.close()
    
    logger.info(f"XAI Visuals saved to {results_dir}")
    logger.info(f"Generated {len(test_indices)} individual event explanations.")

if __name__ == "__main__":
    main()
