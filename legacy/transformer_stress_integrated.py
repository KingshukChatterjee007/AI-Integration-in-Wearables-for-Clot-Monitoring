import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. FEATURE TOKENIZER (Linear Embedding for Num Features)
# =============================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x * self.weight
        x = x + self.bias
        return x

# =============================================================================
# 2. FT-TRANSFORMER MODEL
# =============================================================================

class ClotTransformer(nn.Module):
    def __init__(self, n_features, n_classes=5, d_token=96, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.n_heads = n_heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, n_classes)
        )
        
    def forward(self, x, return_attention=False):
        tokens = self.tokenizer(x)
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, tokens], dim=1)
        
        x = self.transformer(x)
        logits = self.classifier(self.norm(x[:, 0, :]))
        
        if return_attention:
            return logits, x
        return logits

# =============================================================================
# 3. ADVANCED REGULARIZATION: FOCAL LOSS & MIXUP
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================================
# 4. DATA PREP (Balanced v4 Dataset)
# =============================================================================

def load_integrated_data():
    """Loads the CTGAN-balanced integrated dataset."""
    project_root = Path(__file__).parent.parent
    balanced_path = project_root / 'processed_data' / 'integrated_features_balanced_v1.csv'
    
    if not balanced_path.exists():
        raise FileNotFoundError(f"Balanced dataset not found at {balanced_path}. Run ctgan_balancing.py first.")
        
    df = pd.read_csv(balanced_path)
    logger.info(f"Loaded Balanced Dataset: {df.shape} samples")
    
    # Fill missing subject IDs (synthetic data gets its own group)
    df['subject_id'] = df['subject_id'].fillna('synthetic')
    
    # Filter features
    non_features = ['subject_id', 'activity', 'window_id', 'risk_category', 'session', 'timestamp_start', 'target']
    feature_cols = [c for c in df.columns if c in df.columns and c not in non_features]
    
    X = df[feature_cols].copy()
    X = X.fillna(X.median())
    y = df['target'].astype(int).values
    subjects = df['subject_id'].astype(str).values
    
    # Subject-Wise Normalization
    logger.info("Applying subject-wise normalization...")
    X_vals = X.values
    unique_subs = np.unique(subjects)
    logger.info(f"Found {len(unique_subs)} unique subjects: {unique_subs}")
    for sub in unique_subs:
        mask = (subjects == sub)
        m_sum = mask.sum()
        if m_sum > 0:
            # logger.info(f"Normalizing sub: {sub} ({m_sum} samples)")
            sub_scaler = StandardScaler()
            try:
                X_vals[mask] = sub_scaler.fit_transform(X_vals[mask])
            except Exception as e:
                logger.error(f"Failed for sub {sub} with sum {m_sum}. Slice shape: {X_vals[mask].shape}")
                raise e
    
    # Quantile Transformer
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_transformed = qt.fit_transform(X_vals)
    
    class_names = np.array(['Low', 'Low-Moderate', 'Moderate', 'High', 'Critical'])
    
    return X_transformed, y, subjects, class_names, feature_cols

def train_transformer():
    logger.info("Starting Clot-Stress Integrated Transformer Training (v4 Balanced)")
    X, y, subjects, class_names, feature_names = load_integrated_data()
    n_features = X.shape[1]
    
    # Group Shuffle Split (Unseen Subjects)
    gs = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gs.split(X, y, groups=subjects))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = ClotTransformer(n_features=n_features, n_classes=len(class_names)).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Integrated Model Parameters: {param_count:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = FocalLoss(gamma=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    # Class weights for WeightedRandomSampler
    counts = np.bincount(y_train)
    weights = 1.0 / torch.FloatTensor(counts)
    sample_weights = weights[torch.LongTensor(y_train)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), 
                              batch_size=64, sampler=sampler)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), 
                             batch_size=128, shuffle=False)
    
    # Ensure save directory exists
    Path('trained_models').mkdir(exist_ok=True)
    
    best_acc = 0
    epochs = 60
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.4)
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                pred = out.argmax(1)
                total += by.size(0)
                correct += (pred == by).sum().item()
                
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'trained_models/clot_transformer_balanced_best.pth')
            
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}% (Best: {best_acc:.2f}%)")

    # Final Evaluation
    logger.info("\n" + "="*60)
    logger.info("FINAL EVALUATION ON UNSEEN SUBJECTS (v4 Balanced)")
    logger.info("="*60)
    
    model.load_state_dict(torch.load('trained_models/clot_transformer_balanced_best.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            out = model(bx)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            
    from sklearn.metrics import classification_report
    # Ensure all 5 classes are reported even if missing from the small test set
    report = classification_report(all_labels, all_preds, target_names=class_names, labels=[0, 1, 2, 3, 4])
    logger.info(f"\nBalanced Classification Report:\n{report}")
    
    report_path = Path('model_comparison_plots_CLEAN/TRANSFORMER_BALANCED_REPORT.txt')
    report_path.write_text(f"Balanced Clot/Stress Transformer Report (v4)\nTotal Samples: {len(X)}\nFeatures: {n_features}\nModel Params: {param_count:,}\nBest Test Acc: {best_acc:.2f}%\n\n{report}")
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_transformer()
