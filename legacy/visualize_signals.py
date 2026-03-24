import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_signals():
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    
    # We'll plot BVP (feature 0) and EDA (feature 1) as they are most visual
    # Feature 0: BVP, Feature 1: EDA, Feature 2: HR, Feature 3: Temp
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    for i in range(5):
        # Find the first sample of this class
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            sample = X[idx[0]].numpy() # (30, 188)
            time = np.arange(30)
            
            # Plot BVP for this class
            axes[i].plot(time, sample[:, 0], color='red', label='BVP (Heart)')
            axes[i].set_title(f"Class: {class_names[i]}")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylabel("Normalized Signal")
            
            # Sub-plot EDA in secondary axis if wanted, or just BVP
            axes2 = axes[i].twinx()
            axes2.plot(time, sample[:, 1], color='blue', alpha=0.5, label='EDA (Stress)')
            axes2.set_ylabel("EDA", color='blue')
            
        else:
            axes[i].text(0.5, 0.5, f"No samples found for {class_names[i]}", ha='center')

    axes[-1].set_xlabel("Time Steps (Sliding Window)")
    plt.suptitle("Physiological Signal Visualization: Multi-Modal Feature Extraction", fontsize=16)
    
    Path("model_comparison_plots_CLEAN/visualizations").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/ecg_style_signals.png")
    print("Saved signal plot to model_comparison_plots_CLEAN/visualizations/ecg_style_signals.png")

if __name__ == "__main__":
    plot_signals()
