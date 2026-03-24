import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_binary_signals():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    
    # Binary Mapping: 0,1,2 -> Stable; 3,4 -> Risk
    y_bin = np.where(y <= 2, 0, 1)
    class_names = ['Stable (Non-Critical)', 'Risk (Critical)']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    colors = ['#2ECC71', '#E74C3C'] # Green for Stable, Red for Risk
    
    for i in range(2):
        idx = np.where(y_bin == i)[0]
        if len(idx) > 0:
            # Pick a representative sample (median sample or just first)
            sample = X[idx[len(idx)//2]].numpy() # Middle sample of the class
            time = np.arange(30)
            
            # Plot BVP (Feature 0)
            axes[i].plot(time, sample[:, 0], color=colors[i], linewidth=2, label='BVP (Heart Rate)')
            axes[i].set_title(f"Clinical Category: {class_names[i]}", fontsize=14, fontweight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.5)
            axes[i].set_ylabel("Normalized BVP Amplitude")
            
            # Plot EDA (Feature 1) on secondary axis
            axes2 = axes[i].twinx()
            axes2.plot(time, sample[:, 1], color='#3498DB', alpha=0.6, label='EDA (Stress)')
            axes2.set_ylabel("EDA (Skin Conductance)", color='#3498DB')
            
            # Combine legends
            lines, labels = axes[i].get_legend_handles_labels()
            lines2, labels2 = axes2.get_legend_handles_labels()
            axes[i].legend(lines + lines2, labels + labels2, loc='upper right')
            
        else:
            axes[i].text(0.5, 0.5, f"No samples found for {class_names[i]}", ha='center')

    axes[-1].set_xlabel("Time (30-second window steps)", fontsize=12)
    plt.suptitle("Binary Risk Classification: Physiological Fingerprints", fontsize=16, y=0.95)
    
    save_path = "model_comparison_plots_CLEAN/visualizations/binary_ecg_signals.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved binary signal plot to {save_path}")

if __name__ == "__main__":
    plot_binary_signals()
