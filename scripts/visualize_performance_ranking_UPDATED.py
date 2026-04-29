import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Data for the updated ranking (including SWAT Tier)
# Values from MODEL_COMPARISON_REPORT.txt
data = {
    'Model': [
        'SWAT Tier\n(Hybrid)', 'XGBoost', 'Gradient\nBoosting', 'CatBoost', 
        'Random\nForest', 'KNN', 'Decision\nTree', 'Extra\nTrees', 
        'SVM (RBF)', 'Logistic\nReg.', 'AdaBoost', 'Naive Bayes'
    ],
    'F1 Score': [59.97, 83.70, 82.27, 72.11, 71.42, 72.67, 72.67, 69.34, 65.66, 62.44, 56.51, 35.85],
    'Accuracy': [63.52, 84.38, 82.96, 75.36, 75.12, 74.82, 74.70, 73.69, 70.61, 67.16, 64.13, 31.35],
    'Precision': [58.00, 84.08, 82.83, 75.58, 76.32, 73.85, 73.19, 74.81, 69.89, 64.62, 57.53, 55.63],
    'Recall': [63.52, 84.38, 82.96, 75.36, 75.12, 74.82, 74.70, 73.69, 70.61, 67.16, 64.13, 31.35]
}

df = pd.DataFrame(data)

# Set the style to match the user's uploaded image (dark grid, clean bars)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 9), dpi=200)

# Metrics and their corresponding colors from the user's image
# Pink, Golden/Khaki, Green, Teal
metrics = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
colors = ['#F06292', '#AE933F', '#2E7D32', '#00838F']

x = np.arange(len(df['Model']))
width = 0.2

# Plot each metric as a separate bar in the group
for i, metric in enumerate(metrics):
    ax.bar(x + (i - 1.5) * width, df[metric], width, label=metric, color=colors[i], edgecolor='white', linewidth=0.5)

# Formatting
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', labelpad=15)
ax.set_title('Overall Model Performance Ranking (Clean Data - No Leakage)', 
             fontsize=16, fontweight='bold', pad=30, color='#2C3333')
ax.set_xticks(x)
ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.set_ylim(0, 100)
ax.legend(loc='upper right', frameon=True, fontsize=10)

# Aesthetic touches
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Highlight SWAT Tier as the "True SOTA"
ax.annotate('Clinical SOTA\n(Zero-Gap)', xy=(0, 65), xytext=(0, 85),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
            ha='center', fontsize=10, fontweight='bold', color='#1A5F7A')

plt.tight_layout()

# Save image
output_path = Path('reports/overall_performance_ranking_UPDATED.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Updated performance ranking plot successfully saved to: {output_path}")
