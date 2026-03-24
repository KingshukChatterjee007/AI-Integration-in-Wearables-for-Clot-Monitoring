import matplotlib.pyplot as plt
import numpy as np

# Phase Data
phases = ['Phase 1\n(Baseline)', 'Phase 2\n(Transition)', 'Phase 3\n(Integration)', 'Phase 4\n(SWAT-Tier)']
accuracy = [30, 35, 44, 72.11]
precision = [0, 8, 58, 75]
recall = [0, 5, 49, 99]

# Modern, premium style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.edgecolor']='#333333'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333333'
plt.rcParams['ytick.color']='#333333'

def create_plot(data, title, filename, color, ylabel):
    plt.figure(figsize=(10, 6), dpi=150)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Plotting line and markers
    plt.plot(phases, data, marker='o', color=color, linewidth=3, markersize=10, label=title)
    
    # Fill area under curve for modern look
    plt.fill_between(phases, data, color=color, alpha=0.1)
    
    # Annotation of values
    for i, val in enumerate(data):
        plt.annotate(f'{val}%', (phases[i], data[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color=color)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20, color='#1A1A1A')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold', color='#444444')
    plt.ylim(0, 110)
    
    # Aesthetic touches
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Generate the three plots
create_plot(accuracy, "Global Prediction Accuracy", "evolution_accuracy.png", "#2E5BFF", "Accuracy (%)")
create_plot(precision, "Critical Alert Precision (Trust Score)", "evolution_precision.png", "#FF2E63", "Precision (%)")
create_plot(recall, "Critical Case Recall (Safety Score)", "evolution_recall.png", "#08D9D6", "Recall (%)")

print("Graphs generated successfully.")
