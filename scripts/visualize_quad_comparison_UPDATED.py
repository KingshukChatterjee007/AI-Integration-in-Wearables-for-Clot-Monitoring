import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Updated Data (including SWAT Tier) from MODEL_COMPARISON_REPORT.txt
data = {
    'Model': [
        'Naive Bayes', 'AdaBoost', 'SWAT Tier\n(Hybrid)', 'Logistic Reg.', 'SVM (RBF)', 
        'Extra Trees', 'Decision Tree', 'KNN', 'Random Forest', 
        'CatBoost', 'Gradient Boosting', 'XGBoost'
    ],
    'Test Accuracy': [31.35, 64.13, 63.52, 67.16, 70.61, 73.69, 74.70, 74.82, 75.12, 75.36, 82.96, 84.38],
    'F1 Score': [35.85, 56.51, 59.97, 62.44, 65.66, 69.34, 72.67, 72.67, 71.42, 72.11, 82.27, 83.70],
    'Precision': [55.63, 57.53, 58.00, 64.62, 69.89, 74.81, 73.19, 73.85, 76.32, 75.58, 82.83, 84.08],
    'Recall': [31.35, 64.13, 63.52, 67.16, 70.61, 73.69, 74.70, 74.82, 75.12, 75.36, 82.96, 84.38]
}

df = pd.DataFrame(data)

# Modern Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=200)

# Metrics, Titles, and Colors corresponding to the user's image
metrics = ['Test Accuracy', 'F1 Score', 'Precision', 'Recall']
titles = ['Test Accuracy Comparison (Clean Data)', 'F1 Score Comparison', 'Precision Comparison', 'Recall Comparison']
colors = ['#4682B4', '#FF7F50', '#90EE90', '#FFD700']  # Blue, Coral, LightGreen, Gold

for i, ax in enumerate(axes.flatten()):
    metric = metrics[i]
    color = colors[i]
    
    # Sort data for this specific plot if needed (keeping it consistent with user's ascending style)
    # The SWAT Tier is placed logically according to its performance
    bars = ax.barh(df['Model'], df[metric], color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_title(titles[i], fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'{metric} (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    
    # Add values at the end of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                va='center', fontsize=9, fontweight='bold')
        
        # Highlight SWAT Tier bar differently if needed, but let's stick to consistent color for now
        # or add a marker next to it
        if bar.get_y() == df[df['Model'].str.contains('SWAT')].index[0]:
            pass # We could add an arrow here later

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(pad=4.0)

# Save image
output_path = Path('reports/quad_performance_comparison_UPDATED.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Updated quad-panel comparison plot successfully saved to: {output_path}")
