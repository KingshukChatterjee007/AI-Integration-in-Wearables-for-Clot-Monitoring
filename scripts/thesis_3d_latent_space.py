import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def generate_latent_space_plot():
    """
    Academic visualization showing high-dimensional class separation 
    via PCA 3D Latent Space clustering.
    """
    print("--- Generating Thesis Visual #1: 3D Latent Space Clusters ---")
    
    DATA_PATH = 'processed_data/integrated_features_enhanced_CLEAN.csv'
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # 1. Load data
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Identify feature and target columns
    risk_col = 'risk_category'
    features = [c for c in df.columns if c not in [risk_col, 'target', 'split', 'subject_id', 'activity', 'window_id', 'data_source']]
    X = df[features].select_dtypes(include=[np.number])
    y = df[risk_col]
    
    # 2. Pre-processing: Standardization is critical for PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA for 3 Components
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Proportions of variance explained
    var_exp = pca.explained_variance_ratio_
    print(f"Explained Variance: PC1={var_exp[0]:.2%}, PC2={var_exp[1]:.2%}, PC3={var_exp[2]:.2%}")
    
    # 4. Create Plotly Figure
    fig = go.Figure()
    
    colors = {
        'Low': '#2e7d32',          # Deep Green
        'Low-Moderate': '#9ccc65', # Light Green
        'Moderate': '#fbc02d',     # Amber
        'High': '#ef6c00',         # Orange
        'Critical': '#c62828'      # Clinical Red
    }
    
    # Plot each class as a separate trace for better legend control
    for risk_tier in ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']:
        mask = (y == risk_tier)
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                z=X_pca[mask, 2],
                mode='markers',
                marker=dict(
                    size=4, 
                    color=colors[risk_tier],
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=risk_tier
            ))
            
    # 5. Theoretical "Clinical Manifold" (Optional visual guide)
    # Adding a light trace connecting centroids of each cluster to show progression
    centroids = []
    for risk_tier in ['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical']:
        mask = (y == risk_tier)
        if mask.any():
            centroids.append(X_pca[mask].mean(axis=0))
    
    centroids = np.array(centroids)
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='Clinical Risk Manifold'
    ))

    # Layout Tuning
    fig.update_layout(
        title=dict(
            text="<b>Thesis Visual 1: 3D Latent Space Clusters</b><br><sup>Dimensionality Reduction (PCA) Proving Class Separability</sup>",
            font=dict(size=20, family="Times New Roman")
        ),
        scene=dict(
            xaxis_title=f'PC1 ({var_exp[0]:.1%})',
            yaxis_title=f'PC2 ({var_exp[1]:.1%})',
            zaxis_title=f'PC3 ({var_exp[2]:.1%})',
            bgcolor="rgb(250, 250, 250)"
        ),
        legend=dict(title=dict(text='Clinical Risk Tiers'), y=0.9),
        margin=dict(l=0, r=0, b=0, t=100)
    )

    # Save
    os.makedirs('reports/visuals', exist_ok=True)
    fig.write_html('reports/visuals/thesis_v1_latent_space.html')
    print("Visual saved: reports/visuals/thesis_v1_latent_space.html")

if __name__ == "__main__":
    generate_latent_space_plot()
