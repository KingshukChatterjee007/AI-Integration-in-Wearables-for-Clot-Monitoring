import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.clot_hybrid_5class import ClotHybrid5Class

def generate_3d_clusters():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for feature extraction: {device}")

    # 1. Load Data
    test_df = pd.read_csv("processed_data/augmented_5class_test_gen.csv")
    y_test = test_df['target'].values
    drop_cols = ['target', 'split', 'subject_id']
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).values
    n_features = X_test.shape[1]
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=64, shuffle=False)

    # 2. Load Model
    model = ClotHybrid5Class(n_features=n_features, n_classes=5).to(device)
    model.load_state_dict(torch.load("trained_models/clot_5class_hybrid_gen.pth", map_location=device))
    model.eval()

    # 3. Extract Latent Features (Pooled layer)
    # We use a forward hook to capture the 128-dim features before the linear head
    latent_features = []
    
    def hook(module, input, output):
        # The output of 't_out' is pooled in the model's forward
        # However, it's easier to just grab the 'pooled' state if we modify the model slightly
        # Or we can just hook the mean operation. 
        # For simplicity, I'll temporarily wrap the model logic here.
        pass

    print("Extracting 128-dimensional latent representations...")
    with torch.no_grad():
        for xb in test_loader:
            x = xb[0]
            # Standard forward pass logic to get 'pooled'
            batch_size = x.size(0)
            if x.size(1) < model.padded_features:
                padding = torch.zeros(batch_size, model.padded_features - x.size(1), device=x.device)
                x = torch.cat([x, padding], dim=1)
            x = x.view(batch_size, model.seq_len, model.input_size)
            x = x.transpose(1, 2)
            feat = model.cnn(x)
            feat = feat.transpose(1, 2)
            lstm_out, _ = model.lstm(feat)
            t_in = lstm_out + model.pos_encoder
            t_out = model.transformer(t_in)
            pooled = torch.mean(t_out, dim=1)
            latent_features.append(pooled.cpu().numpy())

    latent_matrix = np.vstack(latent_features)

    # 4. Dimension Reduction (128D -> 3D)
    print("Projecting latent space to 3D using PCA...")
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(latent_matrix)

    # 5. Prepare Plotly DataFrame
    class_names = {0: 'Low', 1: 'Low-Moderate', 2: 'Moderate', 3: 'High', 4: 'Critical'}
    df_plot = pd.DataFrame(pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_plot['Risk Tier'] = [class_names[t] for t in y_test]
    
    # 6. Create Premium Midnight Dark Visualization
    fig = px.scatter_3d(
        df_plot, x='PC1', y='PC2', z='PC3',
        color='Risk Tier',
        color_discrete_map={
            'Low': '#00ff00',           # Neon Green
            'Low-Moderate': '#ffff00',  # Neon Yellow
            'Moderate': '#ffa500',      # Orange
            'High': '#ff4500',          # Red-Orange
            'Critical': '#ff0000'       # Pure Red
        },
        opacity=0.7,
        title="3D Latent Space Cluster Cloud: Clot Risk Differentiation",
        labels={'PC1': 'Structural Variance', 'PC2': 'Temporal Signal', 'PC3': 'Relational Intensity'}
    )

    fig.update_traces(marker=dict(size=4, line=dict(width=0, color='DarkSlateGrey')))
    
    # Midnight Premium Theme
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            yaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            zaxis=dict(backgroundcolor="rgb(20, 20, 20)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        font=dict(family="Courier New, monospace", size=14, color="white")
    )

    # Save to HTML
    output_path = "reports/3d_risk_clusters_midnight.html"
    fig.write_html(output_path)
    print(f"SUCCESS: Interactive 3D Latent Space Cloud saved to {output_path}")

if __name__ == "__main__":
    generate_3d_clusters()
