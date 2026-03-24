import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_non_zero_sample(X, y, target_class):
    indices = np.where(y == target_class)[0]
    if len(indices) == 0:
        return None
    
    # Search for a sample with non-zero activity
    for idx in indices:
        sample = X[idx].numpy()
        # Check standard deviation of first 4 features (BVP, EDA, HR, Temp)
        # If std > 0.01, it's likely a real signal
        if np.std(sample[:, :4]) > 1e-3:
            return sample
    
    # Fallback to first sample if no high-std sample found
    return X[indices[0]].numpy()

def generate_visual_1_table():
    print("Generating Visual 1: 5x4 Sensor Grid...")
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    sensor_names = ['BVP Signal', 'HR Signal', 'EDA Signal', 'TEMP Signal']
    colors = ['#2ECC71', '#E74C3C', '#3498DB', '#F1C40F']
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    plt.style.use('dark_background') # Match Image 1 style
    
    for i, cls in enumerate(class_names):
        sample = get_non_zero_sample(X, y, i)
        for j in range(4):
            ax = axes[i, j]
            if sample is not None:
                ax.plot(sample[:, j], color=colors[j], linewidth=2)
                ax.grid(True, color='red', alpha=0.1, linestyle=':')
            else:
                ax.text(0.5, 0.5, "Empty", ha='center')
            
            if i == 0: ax.set_title(sensor_names[j], fontsize=12, fontweight='bold')
            if j == 0: ax.set_ylabel(cls, rotation=0, labelpad=40, fontsize=12, fontweight='bold')
            
    plt.suptitle("Phase 5.1 Hybrid: Physiological Sensor Waveforms (30-Window Temporal Evolution)", fontsize=18, y=0.95)
    Path("model_comparison_plots_CLEAN/visualizations").mkdir(parents=True, exist_ok=True)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/waveform_grid_FIXED.png", facecolor='black')

def generate_visual_2_binary():
    print("Generating Visual 2: Binary Overlay...")
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    
    # Binary Mapping: 0,1,2 -> Stable; 3,4 -> Risk
    y_bin = np.where(y <= 2, 0, 1)
    class_names = ['Stable (Non-Critical)', 'Risk (Critical)']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.style.use('default')
    
    for i in range(2):
        indices = np.where(y_bin == i)[0]
        # Search for non-zero in this group
        sample = None
        for idx in indices:
            s = X[idx].numpy()
            if np.std(s[:, :2]) > 0.05: # High std for BVP/EDA
                sample = s
                break
        if sample is None: sample = X[indices[0]].numpy()
        
        ax = axes[i]
        ax.plot(sample[:, 0], color='#2ECC71', linewidth=2, label='BVP (Heart Rate)')
        ax2 = ax.twinx()
        ax2.plot(sample[:, 1], color='#3498DB', alpha=0.6, label='EDA (Stress)')
        ax.set_title(f"Clinical Category: {class_names[i]}", fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Binary Risk Classification: Physiological Fingerprints", fontsize=16)
    plt.savefig("model_comparison_plots_CLEAN/visualizations/binary_overlay_FIXED.png")

def generate_visual_3_multimodal():
    print("Generating Visual 3: 5-Class Multimodal Overlay...")
    data_dir = Path("processed_data/v5_tensors")
    X = torch.load(data_dir / "X_v5_30seq.pt")
    y = torch.load(data_dir / "y_v5_30seq.pt").numpy()
    
    class_names = ['Low', 'Low-Mod', 'Mod', 'High', 'Critical']
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    for i, cls in enumerate(class_names):
        sample = get_non_zero_sample(X, y, i)
        ax = axes[i]
        if sample is not None:
            ax.plot(sample[:, 0], color='brown', label='Signal')
            ax2 = ax.twinx()
            ax2.plot(sample[:, 1], color='blue', alpha=0.4, label='EDA')
            ax.set_title(f"Class: {cls}")
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("model_comparison_plots_CLEAN/visualizations/ecg_style_signals_FIXED.png")

if __name__ == "__main__":
    generate_visual_1_table()
    generate_visual_2_binary()
    generate_visual_3_multimodal()
