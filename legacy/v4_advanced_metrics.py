import torch
import pandas as pd
import numpy as np
from transformer_stress_integrated import ClotTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics.pairwise import cosine_similarity

def calculate_advanced_metrics(model_path, data_path):
    df = pd.read_csv(data_path)
    numeric_df = df.select_dtypes(include=[np.number]).fillna(0.0)
    X = numeric_df.drop(columns=['target', 'window_id'], errors='ignore').values
    y = numeric_df['target'].values
    
    # Same split as training
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    n_features = X_val_scaled.shape[1]
    model = ClotTransformer(n_features=n_features, n_classes=5).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val_scaled))
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=1)
        
    # 1. F1 Score
    f1_weighted = f1_score(y_val, preds, average='weighted')
    f1_macro = f1_score(y_val, preds, average='macro')
    
    # 2. ROC-AUC (One-vs-Rest)
    y_val_bin = label_binarize(y_val, classes=[0,1,2,3,4])
    roc_auc = roc_auc_score(y_val_bin, probs, multi_class='ovr', average='weighted')
    
    # 3. Cosine Similarity (Representative measure: Avg similarity between predicted and true feature centroids)
    # Or simply similarity between sample features of same class
    cos_sim_list = []
    for cls in range(5):
        mask = (y_val == cls)
        if mask.any():
            feats = X_val[mask]
            # Average similarity within the class features
            if len(feats) > 1:
                sim = cosine_similarity(feats)
                avg_sim = (np.sum(sim) - len(feats)) / (len(feats) * (len(feats) - 1))
                cos_sim_list.append(avg_sim)
    avg_feature_cos_sim = np.mean(cos_sim_list)

    print("=== Phase 4 Advanced Performance Metrics ===")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro):    {f1_macro:.4f}")
    print(f"ROC-AUC (Weighted):  {roc_auc:.4f}")
    print(f"Avg Feature Cosine Similarity (Intra-class): {avg_feature_cos_sim:.4f}")

if __name__ == "__main__":
    data_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\processed_data\integrated_features_v4_TEMPORAL.csv"
    model_file = r"c:\Users\91704\AI-Integration-in-Wearables-for-Clot-Monitoring\trained_models\clot_transformer_V4_OPTIMIZED.pth"
    calculate_advanced_metrics(model_file, data_file)
