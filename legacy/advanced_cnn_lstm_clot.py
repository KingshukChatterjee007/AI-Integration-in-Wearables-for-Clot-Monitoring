"""
Robust 1D CNN + Bi-LSTM for Blood Clot Risk Classification
==========================================================

ANTI-OVERFITTING ARCHITECTURE:
- Subject-Grouped Splitting (Validation on UNSEEN patients)
- Biometric Exclusion (Removed Age, BMI, Weight to prevent leakage)
- PyTorch Native Multi-Head Attention (4 heads)
- Input Gaussian Noise injection
- Weight Decay 1e-3 + Label Smoothing 0.1

Target: zero-variation generalization across patients.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import copy
import time
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
DEVICE = torch.device('cpu')

plt.style.use('seaborn-v0_8-darkgrid')


# =============================================================================
# STEP 1: DATA LOADING WITH HEALTH CHECKS
# =============================================================================

def load_and_prepare_data():
    """Load clean dataset with biometric exclusion and leakage checks."""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DATA LOADING & HEALTH CHECKS")
    logger.info("="*60)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'processed_data' / 'integrated_features_enhanced_CLEAN.csv'

    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Subject Downsampling (Fair Weighting) ---
    # Subjects s10, s11, s12 have 6x more data (860+ windows). 15 subjects have ~145.
    # We cap at 200 windows to balance the training influence of each person.
    logger.info(f"   Downsampling large subjects to cap of 200 windows...")
    sampled_dfs = []
    for sub in df['subject_id'].unique():
        sub_df = df[df['subject_id'] == sub]
        if len(sub_df) > 200:
            sampled_dfs.append(sub_df.sample(n=200, random_state=42))
        else:
            sampled_dfs.append(sub_df)
    df = pd.concat(sampled_dfs).reset_index(drop=True)
    logger.info(f"   New Balanced Data size: {df.shape[0]:,} rows")

    # Exclude non-feature columns AND biometric IDs
    # Biesthetics like Age, BMI, Weight, Height, Gender allow the model to cheat by identifying individuals
    non_feature_cols = ['subject_id', 'activity', 'window_id', 'risk_category']
    biometric_cols = [
        'age', 'weight', 'bmi', 'height', 'gender_encoded', 
        'data_source', 'subject_id'
    ]
    
    feature_cols = [c for c in df.columns if c not in non_feature_cols and c.lower() not in biometric_cols]
    
    # Store subjects separately for splitting
    subjects = df['subject_id'].values
    X_raw = df[feature_cols].select_dtypes(include=[np.number])
    y_raw = df['risk_category']

    # Health check: check for leakage labels
    leaky_keywords = ['composite_risk', 'risk_score', 'risk_level', 'label', 'target']
    leaky_cols = [c for c in X_raw.columns if any(k in c.lower() for k in leaky_keywords)]
    if leaky_cols:
        logger.warning(f"   REMOVING {len(leaky_cols)} potentially leaky columns: {leaky_cols}")
        X_raw = X_raw.drop(columns=leaky_cols)

    # Missing values
    X_clean = X_raw.fillna(X_raw.median())

    # Low variance
    variances = X_clean.var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        X_clean = X_clean.drop(columns=low_var_cols)

    logger.info(f"   ✓ Biometric features removed for leakage prevention")
    logger.info(f"   Final feature count: {X_clean.shape[1]}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    class_names = le.classes_

    X_data = X_clean.values
    y_data = y_enc

    # --- Subject-Wise Normalization ---
    # Standardize each subject's data independently to remove individual baseline offsets
    logger.info(f"   Applying subject-wise signal normalization...")
    for sub in np.unique(subjects):
        mask = (subjects == sub)
        sub_scaler = StandardScaler()
        X_data[mask] = sub_scaler.fit_transform(X_data[mask])

    return X_data, y_data, subjects, class_names, X_clean.shape[1]


# =============================================================================
# STEP 2: CONSERVATIVE AUGMENTATION
# =============================================================================

def augment_training_data(X_train, y_train):
    """
    Conservative augmentation:
    - SMOTE-Tomek balances classes (with boundary cleaning)
    - Jitter added ~1x (not 2x/3x) to prevent memorization
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: CONSERVATIVE AUGMENTATION")
    logger.info("="*60)
    logger.info(f"   Original training set: {len(y_train):,} samples")

    # SMOTE-Tomek
    class_counts = np.bincount(y_train)
    k_neighbors = min(5, np.min(class_counts[class_counts > 0]) - 1)
    k_neighbors = max(1, k_neighbors)

    try:
        smt = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors, random_state=42))
        X_res, y_res = smt.fit_resample(X_train, y_train)
    except Exception as e:
        logger.warning(f"   SMOTE-Tomek failed ({e}), using SMOTE")
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    logger.info(f"   After SMOTE-Tomek: {len(y_res):,} samples")
    logger.info(f"   Distribution: {dict(zip(range(5), np.bincount(y_res)))}")

    # Conservative jitter ONCE only (not 2x or more)
    noise = np.random.normal(0, 0.005, X_res.shape)          # σ = 0.005 (very small)
    scale = np.random.uniform(0.97, 1.03, (X_res.shape[0], 1))  # ±3% only
    X_jitter = (X_res * scale) + noise
    X_aug = np.vstack([X_res, X_jitter])
    y_aug = np.concatenate([y_res, y_res])

    logger.info(f"   After gentle jitter (+1x): {len(y_aug):,} samples")
    logger.info(f"   ✓ Augmentation ratio: {len(y_aug)/len(y_train):.1f}x (kept conservative)")
    return X_aug, y_aug


# =============================================================================
# STEP 3: MODEL WITH STRONG REGULARIZATION & MULTI-HEAD ATTENTION
# =============================================================================

class GaussianNoise(nn.Module):
    """Gaussian noise injection layer for robust training"""
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.stddev
        return x


class CNN_BiLSTM_V2(nn.Module):
    """
    ULTRA-Regularized 1D CNN + Bi-LSTM:
    - Input Gaussian Noise injection
    - Reduced capacity (16/32 CNN, 32 Bi-LSTM)
    - Dropout 0.5 everywhere
    """
    def __init__(self, n_features, n_classes=5, dropout=0.5):
        super().__init__()

        self.noise = GaussianNoise(0.01)

        # -- 1D CNN Path --
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)  # Reduced filters to 8
        self.bn1   = nn.BatchNorm1d(8)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(8, 8, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(8)
        self.drop2 = nn.Dropout(dropout)

        self.pool  = nn.AdaptiveAvgPool1d(16)  # Reduced seq length to 16

        # -- Bi-LSTM Path --
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=8,          # 16 bidirectional
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.drop_lstm = nn.Dropout(dropout)

        # -- Multi-Head Attention --
        self.attention = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )

        # -- Classifier Head --
        self.classifier = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.7),        # Stronger dropout
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        # x: (B, features)
        x = self.noise(x)
        
        # CNN
        x = x.unsqueeze(1)                   # (B, 1, features)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.drop2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(x)                     # (B, 8, 16)

        # LSTM
        x = x.transpose(1, 2)               # (B, 16, 8)
        lstm_out, _ = self.lstm(x)          # (B, 16, 16)
        lstm_out = self.drop_lstm(lstm_out)

        # Multi-Head Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Pool over time
        out = attn_out.mean(dim=1)           # (B, 16)

        return self.classifier(out)


# =============================================================================
# STEP 4: TRAINING WITH LR SCHEDULER + GRADIENT CLIPPING
# =============================================================================

def train_model(X_train, y_train, X_val, y_val, class_names):
    logger.info("\n" + "="*60)
    logger.info("STEP 4: TRAINING WITH ANTI-OVERFITTING CONTROLS")
    logger.info("="*60)

    n_classes = len(class_names)
    model = CNN_BiLSTM_V2(n_features=X_train.shape[1], n_classes=n_classes, dropout=0.5).to(DEVICE)
    logger.info(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Label smoothing prevents overconfident class probability → reduces overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)

    # Balanced L2 weight decay (1e-3) and slightly higher LR (1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # LR Scheduler: reduce LR when val acc plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=12, factor=0.5, min_lr=1e-6
    )

    # Balanced weighted sampler
    counts = np.bincount(y_train)
    weights = 1.0 / torch.FloatTensor(counts)
    sample_wts = weights[torch.LongTensor(y_train)]
    sampler = WeightedRandomSampler(sample_wts, len(sample_wts))

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds   = TensorDataset(torch.FloatTensor(X_val),   torch.LongTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    best_acc = 0.0
    best_state = None
    patience_counter = 0
    max_patience = 20
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': [], 'lr': []}

    logger.info(f"   Training config: LR={optimizer.param_groups[0]['lr']}, L2={optimizer.param_groups[0]['weight_decay']}, Dropout=0.5, LabelSmoothing=0.15")
    logger.info(f"   Gradient clipping: max_norm=1.0")

    for epoch in range(200):
        # ---- Train ----
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            # Gradient clipping prevents exploding gradients → stable training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
            t_correct += out.argmax(1).eq(yb).sum().item()
            t_total += yb.size(0)

        # ---- Validate ----
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(DEVICE), yv.to(DEVICE)
                ov = model(xv)
                v_loss += criterion(ov, yv).item()
                v_correct += ov.argmax(1).eq(yv).sum().item()
                v_total += yv.size(0)

        train_acc = t_correct / t_total
        val_acc   = v_correct / v_total
        train_loss = t_loss / len(train_loader)
        val_loss   = v_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Scheduler step (based on val_acc)
        scheduler.step(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            gap = (train_acc - val_acc) * 100
            logger.info(
                f"   Epoch {epoch+1:3d} | "
                f"Train: {train_acc*100:5.2f}% | "
                f"Val: {val_acc*100:5.2f}% | "
                f"Gap: {gap:+.2f}% | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"   ⟳ Early stopping at epoch {epoch+1}. Best val acc: {best_acc*100:.2f}%")
                break

    model.load_state_dict(best_state)
    logger.info(f"\n   ✓ Best Model Val Accuracy: {best_acc*100:.2f}%")
    return model, history, best_acc


# =============================================================================
# STEP 5: FINAL EVALUATION AND REPORTING
# =============================================================================

def evaluate_and_save(model, X_test, y_test, class_names, history, project_root):
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test)).argmax(1).numpy()

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')
    report = classification_report(y_test, preds, target_names=class_names)
    cm = confusion_matrix(y_test, preds)

    logger.info("\n" + "="*60)
    logger.info(f"FINAL TEST ACCURACY: {acc*100:.2f}%  |  F1: {f1*100:.2f}%")
    logger.info("="*60)
    logger.info(f"\nClassification Report:\n{report}")

    # Overfitting check
    final_train_acc = history['train_acc'][-1]
    final_val_acc   = history['val_acc'][-1]
    gap = (final_train_acc - final_val_acc) * 100
    logger.info(f"   Final Gap (Train-Val): {gap:+.2f}%")
    if gap < 3:
        logger.info("   ✅ VERDICT: Well generalizing — no significant overfitting")
    elif gap < 8:
        logger.info("   ⚠️  VERDICT: Slight overfitting — acceptable range")
    else:
        logger.warning("   ❌ VERDICT: Significant overfitting detected!")

    plots_dir = project_root / 'model_comparison_plots_CLEAN'
    plots_dir.mkdir(exist_ok=True)

    # Save model
    torch.save(model.state_dict(), project_root / 'trained_models' / 'advanced_cnn_lstm_v2.pth')

    # Fig 1: Training curves (separate train/val for gap visibility)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs = range(1, len(history['train_acc']) + 1)

    axes[0].plot(epochs, [a*100 for a in history['train_acc']], label='Train', color='steelblue')
    axes[0].plot(epochs, [a*100 for a in history['val_acc']],   label='Val',   color='coral')
    axes[0].fill_between(epochs,
                          [a*100 for a in history['train_acc']],
                          [a*100 for a in history['val_acc']],
                          alpha=0.2, color='red', label='Gap')
    axes[0].set_title('Accuracy (Gap Highlighted)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy %')
    axes[0].legend()

    axes[1].plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
    axes[1].plot(epochs, history['val_loss'],   label='Val Loss',   color='coral')
    axes[1].set_title('Loss Curves')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(epochs, history['lr'], color='purple')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(plots_dir / '12_cnn_lstm_v2_curves.png', dpi=150)
    plt.close()

    # Fig 2: Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
                cmap='Purples', ax=ax)
    ax.set_title(f'1D CNN + Bi-LSTM v2 — Test Accuracy: {acc*100:.2f}%')
    plt.tight_layout()
    plt.savefig(plots_dir / '13_cnn_lstm_v2_cm.png', dpi=150)
    plt.close()

    # Save Text Report
    report_text = f"""
================================================================================
1D CNN + Bi-LSTM v2 — ANTI-OVERFITTING REDESIGN
AI Integration in Wearables for Blood Clot Monitoring
================================================================================
Generated: {__import__('datetime').datetime.now()}

ANTI-OVERFITTING MEASURES APPLIED:
  ✓ Label Smoothing (0.1) — prevents overconfident predictions
  ✓ Dropout 0.5 — at every layer boundary
  ✓ L2 Weight Decay (1e-3) — 10x stronger than previous
  ✓ Gradient Clipping (max_norm=1.0)
  ✓ Conservative Augmentation (1x jitter, σ=0.005 noise only)
  ✓ ReduceLROnPlateau scheduler (patience=8, factor=0.5)
  ✓ Label health checks (leakage, low-variance feature removal)

FINAL RESULTS:
  Test Accuracy:              {acc*100:.2f}%
  Weighted F1 Score:          {f1*100:.2f}%
  Final Train vs Val Gap:     {gap:+.2f}%  ← should be < 5%

PER-CLASS PERFORMANCE:
{report}
================================================================================
"""
    (plots_dir / 'CNN_LSTM_V2_REPORT.txt').write_text(report_text, encoding='utf-8')
    logger.info(f"\n   Report saved to: {plots_dir / 'CNN_LSTM_V2_REPORT.txt'}")
    logger.info(f"   Plots saved to : {plots_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    logger.info("\n" + "="*60)
    logger.info("SUBJECT-GROUPED 1D CNN + Bi-LSTM v2")
    logger.info("="*60)

    project_root = Path(__file__).parent.parent

    # 1. Load data and subject IDs
    X, y, subjects, class_names, n_features = load_and_prepare_data()

    # 2. Subject-Grouped Split: Ensure patient isolation
    # Test set: ~15% of subjects
    gs_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gs_test.split(X, y, groups=subjects))
    
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    sub_train_full, sub_test = subjects[train_idx], subjects[test_idx]

    # Val set: ~15% of REMAINING subjects
    gs_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gs_val.split(X_train_full, y_train_full, groups=sub_train_full))
    
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
    
    logger.info(f"\n   Subject Isolation Stats:")
    logger.info(f"   - Training Subjects  : {len(np.unique(sub_train_full[train_idx]))}")
    logger.info(f"   - Validation Subjects: {len(np.unique(sub_train_full[val_idx]))}")
    logger.info(f"   - Test Subjects      : {len(np.unique(sub_test))}")
    logger.info(f"   Samples: Train={len(y_train):,}, Val={len(y_val):,}, Test={len(y_test):,}")

    # 3. Scaling (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # 4. Conservative Augmentation on Training data ONLY
    X_aug, y_aug = augment_training_data(X_train_s, y_train)

    # 5. Train
    model, history, best_acc = train_model(X_aug, y_aug, X_val_s, y_val, class_names)

    # 6. Final evaluation on UNSEEN subjects
    evaluate_and_save(model, X_test_s, y_test, class_names, history, project_root)

    logger.info(f"\n   Total time: {(time.time()-t0)/60:.1f} minutes")
    logger.info("   DONE!")

if __name__ == "__main__":
    main()
