import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from clot_hybrid_v5 import ClotHybridV5
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Colors for professional thesis look
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
CLASS_NAMES = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Generates a 5x5 Normalized Confusion Matrix Heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    # Normalize by row (True Class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle division by zero
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix (Subject-Wise Split)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Confusion Matrix to {output_path}")

def plot_pr_bar_chart(y_true, y_pred, output_path):
    """Generates a grouped bar chart for Precision and Recall per class."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, 
                                   labels=[0,1,2,3,4], output_dict=True, zero_division=0)
    
    precision = [report[cls]['precision'] for cls in CLASS_NAMES]
    recall = [report[cls]['recall'] for cls in CLASS_NAMES]
    
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, precision, width, label='Precision', color='#1f77b4', alpha=0.8)
    rects2 = ax.bar(x + width/2, recall, width, label='Recall', color='#ff7f0e', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Precision and Recall (Phase 5.1 Hybrid)')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR Bar Chart to {output_path}")

def plot_phase_evolution(output_path):
    """Generates Phase Evolution Chart (Phases 1 - 5.1)."""
    phases = ['v1 (CNN-LSTM)', 'v2 (Transformer)', 'v3 (Integrated)', 'v4 (Optimized)', 'v5.1 (Hybrid)']
    # Data from MODEL_EVOLUTION_SUMMARY.md mixed with Phase 5.1 results
    accuracy = [30.0, 35.0, 44.3, 62.1, 71.0]
    critical_recall = [0.0, 5.0, 50.0, 68.0, 91.0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(phases, accuracy, marker='o', label='Global Accuracy (%)', linewidth=2, color='#1f77b4')
    plt.plot(phases, critical_recall, marker='s', label='Critical Recall (%)', linewidth=2, color='#d62728')
    
    plt.title('Clot-Monitoring Model Evolution (Phases 1-5.1)')
    plt.ylabel('Percentage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Phase Evolution Chart to {output_path}")

def plot_roc_curve(y_true, y_probs, output_path):
    """Generates ROC curves for each class."""
    plt.figure(figsize=(10, 8))
    for i in range(len(CLASS_NAMES)):
        # One-vs-Rest ROC
        y_true_binary = (y_true == i).astype(int)
        y_score = y_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) per Class')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC Curve to {output_path}")

def plot_reliability_diagram(y_true, y_probs, output_path):
    """Generates a Reliability Diagram to check Bayesian calibration."""
    plt.figure(figsize=(10, 8))
    
    # We focus on the 'Critical' class (index 4) as it's most important
    y_true_binary = (y_true == 4).astype(int)
    prob_true, prob_pred = calibration_curve(y_true_binary, y_probs[:, 4], n_bins=10)
    
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Critical Class (Phase 5.1 Hybrid)')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram: Calibration Audit (Critical Class)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Reliability Diagram to {output_path}")

def generate_thesis_plots():
    data_dir = Path("processed_data/v5_tensors")
    model_path = Path("trained_models/clot_hybrid_v5_best.pth")
    output_dir = Path("model_comparison_plots_CLEAN/thesis_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not (data_dir.exists() and model_path.exists()):
        logger.error("Missing model or data. Run training first.")
        return
        
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt")
    s = torch.load(data_dir / "subjects_v5_30seq.pt")
    
    # Use same split as training
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups=s))
    
    X_test, y_test = X[test_idx], y[test_idx]
    
    model = ClotHybridV5(n_features=X_test.shape[2])
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = torch.argmax(logits, dim=-1).numpy()
    
    # Must-haves
    plot_confusion_matrix(y_test.numpy(), preds, output_dir / "confusion_matrix_v5.png")
    plot_pr_bar_chart(y_test.numpy(), preds, output_dir / "precision_recall_v5.png")
    
    # Should-haves (Evolution)
    plot_phase_evolution(output_dir / "phase_evolution.png")
    
    # Nice-to-haves
    plot_roc_curve(y_test.numpy(), probs, output_dir / "roc_curves_v5.png")
    plot_reliability_diagram(y_test.numpy(), probs, output_dir / "reliability_v5.png")
    
    # Training Curve (if history exists)
    hist_path = Path("trained_models/v5_training_history.csv")
    if hist_path.exists():
        df = pd.read_csv(hist_path)
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='#1f77b4')
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='#ff7f0e')
        plt.title('Training vs Validation Accuracy (Phase 5.1)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "learning_curve_v5.png", dpi=300)
        plt.close()
        logger.info(f"Saved Learning Curve to {output_dir / 'learning_curve_v5.png'}")

if __name__ == "__main__":
    generate_thesis_plots()
