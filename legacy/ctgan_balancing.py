import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.evaluation.single_table import evaluate_quality
import logging

# 1. STRICT CPU ENFORCEMENT
# Forcing PyTorch to CPU to prevent crashes on Intel Iris Xe
torch.set_default_device('cpu')
DEVICE = 'cpu'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing CTGAN Balancing Pipeline (CPU-Only Mode)")
    
    # 2. LOAD MASTER DATASET
    # We load both clinical and stress data to create the 'Integrated Master'
    master_path = Path('processed_data/integrated_features_balanced_v1_TEMP.csv') 
    
    # We first reconstruct the integrated dataset used by the transformer
    clinical_df = pd.read_csv('processed_data/integrated_features_enhanced_CLEAN.csv')
    stress_df = pd.read_csv('processed_data/stress_features_v1.csv')
    
    # 2.1 HARMONIZE FEATURE NAMES (PPG and ACC)
    rename_map = {
        'pleth_mean': 'bvp_mean', 'pleth_std': 'bvp_std', 
        'pleth_min': 'bvp_min', 'pleth_max': 'bvp_max',
        'a_x_var': 'acc_x_var', 'a_y_var': 'acc_y_var', 'a_z_var': 'acc_z_var'
    }
    clinical_df.rename(columns=rename_map, inplace=True)
    
    # 2.2 Mapping Clinical labels correctly
    # Values found: 'Low', 'Low-Moderate', 'Moderate', 'High', 'Critical'
    clinical_df['target'] = clinical_df['risk_category'].map({
        'Low': 0, 'Low-Moderate': 1, 'Moderate': 2, 'High': 3, 'Critical': 4
    })
    logger.info(f"Clinical target counts:\n{clinical_df['target'].value_counts(dropna=False)}")
    
    # 2.3 Mapping Stress Session to Target
    stress_df['target'] = stress_df['session'].map({
        'Final': 4, 'Midterm 1': 3, 'Midterm 2': 3
    })
    logger.info(f"Stress target counts:\n{stress_df['target'].value_counts(dropna=False)}")
    
    # Ensure all columns from both are preserved
    df = pd.concat([clinical_df, stress_df], axis=0, ignore_index=True, sort=False)
    logger.info(f"Combined shape before dropna: {df.shape}")
    
    df = df.dropna(subset=['target'])
    logger.info(f"Combined shape after dropna: {df.shape}")
    
    # Impute missing features
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'activity', 'window_id', 'risk_category', 'session', 'timestamp_start', 'target']]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    logger.info(f"Master dataset size: {len(df)} samples")
    logger.info(f"Final REAL Class distribution:\n{df['target'].value_counts().sort_index()}")
    
    # 3. METADATA CONSTRUCTION
    # Automatically detect types for continuous (EDA/BVP) and categorical (target)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # 4. MULTI-MODEL TARGETED TRAINING
    logger.info("Executing Divide-and-Conquer Multi-GAN Balancing (Classes 1-4)")
    
    counts = df['target'].value_counts()
    target_count = 2000 # Uniform target for each class
    synthetic_pieces = []
    
    for cls in [1, 2, 3, 4]:
        logger.info(f"--- Processing Class {cls} ---")
        real_cls_data = df[df['target'] == cls]
        current_count = len(real_cls_data)
        needed = target_count - current_count
        
        if needed <= 0:
            logger.info(f"Class {cls} already has {current_count} samples. Skipping synthesis.")
            continue
            
        logger.info(f"Training specialized CTGAN for Class {cls} on {current_count} samples...")
        
        # Fit a specialized model for just this class
        # Fewer epochs are needed for a single-class distribution
        cls_synthesizer = CTGANSynthesizer(
            metadata,
            enforce_rounding=False,
            epochs=150, 
            batch_size=min(100, current_count),
            cuda=False
        )
        cls_synthesizer.fit(real_cls_data)
        
        logger.info(f"Generating {needed} high-fidelity samples for Class {cls}...")
        piece = cls_synthesizer.sample(num_rows=needed)
        synthetic_pieces.append(piece)
        
    # Combine all synthetic data
    if synthetic_pieces:
        synthetic_data = pd.concat(synthetic_pieces, ignore_index=True)
    else:
        synthetic_data = pd.DataFrame(columns=df.columns)
    
    # 7. QUALITY ASSURANCE GATE
    if not synthetic_data.empty:
        logger.info("Evaluating Global Multi-GAN Quality...")
        quality_report = evaluate_quality(
            real_data=df[df['target'].isin([1, 2, 3, 4])],
            synthetic_data=synthetic_data,
            metadata=metadata
        )
        quality_score = quality_report.get_score()
        logger.info(f"=== MULTI-GAN TOTAL QUALITY SCORE: {quality_score:.4f} ===")
    
    # 8. DATASET MERGE & FINAL SAVE
    balanced_df = pd.concat([df, synthetic_data], ignore_index=True)
    final_path = Path('processed_data/integrated_features_balanced_v2.csv')
    balanced_df.to_csv(final_path, index=False)
    
    logger.info(f"Globally Balanced dataset (v2) saved to {final_path}")
    logger.info(f"New total samples: {len(balanced_df)}")
    logger.info(f"Final class distribution:\n{balanced_df['target'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()
