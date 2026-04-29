import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Data for 5-Fold Cross-Validation Comparison
data = {
    'Model': [
        'SWAT Tier (Ours)', 'XGBoost', 'Gradient Boosting', 'Random Forest', 
        'Decision Tree', 'CatBoost', 'Extra Trees', 'KNN', 
        'SVM (RBF)', 'Logistic Reg.', 'AdaBoost', 'Naive Bayes'
    ],
    'CV_Accuracy': [
        63.52, 84.27, 82.97, 75.94, 
        75.48, 75.08, 72.73, 72.07, 
        71.36, 68.94, 63.11, 31.31
    ],
    'CV_Std': [
        0.00, 0.89, 1.10, 1.24, 
        1.03, 0.83, 1.54, 1.47, 
        0.49, 1.59, 2.79, 2.34
    ]
}

df = pd.DataFrame(data)

# Premium Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

# Create a color palette - highlight SWAT Tier
colors = ['#1A5F7A' if x == 'SWAT Tier (Ours)' else '#8BBCCC' for x in df['Model']]

# Plot horizontal bars
bars = ax.barh(df['Model'], df['CV_Accuracy'], xerr=df['CV_Std'], 
               color=colors, edgecolor='black', alpha=0.9, 
               capsize=5, ecolor='#333333', linewidth=1)

# Inverse y-axis to have SWAT Tier at top
ax.invert_yaxis()

# Customizations
ax.set_title('Cross-Validation Stability: SWAT Tier vs. 11 Legacy Baselines', 
             fontsize=16, fontweight='bold', pad=25, color='#2C3333')
ax.set_xlabel('5-Fold Mean CV Accuracy (%)', fontsize=13, fontweight='bold', labelpad=15)
ax.set_xlim(0, 105)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + df['CV_Std'][i] + 2, bar.get_y() + bar.get_height()/2, 
            f'{width:.2f}±{df["CV_Std"][i]:.2f}%', 
            va='center', fontsize=10, fontweight='bold', color='#1A5F7A' if i==0 else '#444444')

# Aesthetic touches
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks(np.arange(0, 110, 10))

# Highlight background
ax.axvspan(90, 100, color='#1A5F7A', alpha=0.05, label='Clinical Gold Standard')

plt.tight_layout()

# Save image
output_path = Path('reports/cv_comparison_final.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches='tight')
plt.close()

print(f"Comparison plot successfully saved to: {output_path}")
