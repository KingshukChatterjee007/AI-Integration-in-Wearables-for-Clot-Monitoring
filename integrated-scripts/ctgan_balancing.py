import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
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
    
    # 2.2 Mapping Clinical Activity to Target
    clinical_df['target'] = clinical_df['activity'].map({
        'Basal': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4
    })
    
    # 2.3 Mapping Stress Session to Target
    stress_df['target'] = stress_df['session'].map({
        'Final': 4, 'Midterm 1': 3, 'Midterm 2': 3
    })
    
    # Ensure EDA columns are imputed for clinical data (it didn't have EDA)
    eda_cols = [c for c in stress_df.columns if 'eda' in c]
    for c in eda_cols:
        if c not in clinical_df.columns:
            clinical_df[c] = 0.0 # Baseline
    
    # Ensure columns match for concatenation
    common_cols = list(set(clinical_df.columns) & set(stress_df.columns))
    df = pd.concat([clinical_df[common_cols], stress_df[common_cols]], ignore_index=True)
    df = df.dropna(subset=['target'])
    
    logger.info(f"Master dataset size: {len(df)} samples")
    logger.info(f"Class distribution:\n{df['target'].value_counts()}")
    
    # 3. METADATA CONSTRUCTION
    # Automatically detect types for continuous (EDA/BVP) and categorical (target)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    # 4. TARGETED TRAINING
    # Isolate Minority Classes: "Critical Risk" (4) and "High Risk" (3)
    # Mapping assumed from earlier scripts: 0=Low, 1=Low-Mod, 2=Mod, 3=High, 4=Critical
    minority_data = df[df['target'].isin([3, 4])]
    
    if len(minority_data) == 0:
        logger.error("No samples found for 'Critical Risk' or 'High Risk'. Check target encoding.")
        return

    logger.info(f"Training targeted CTGAN on {len(minority_data)} minority samples...")
    
    # Configuration for manageable CPU load
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=False,
        epochs=300,
        batch_size=500,
        cuda=False # FORCE CPU
    )
    
    # 5. TRAIN SYNTHESIZER
    synthesizer.fit(minority_data)
    
    # 6. DATA GENERATION
    logger.info("Generating 5,000 synthetic minority samples...")
    synthetic_data = synthesizer.sample(num_rows=5000)
    
    # 7. QUALITY ASSURANCE GATE
    logger.info("Evaluating Synthetic Data Quality...")
    quality_report = evaluate_quality(
        real_data=minority_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )
    
    quality_score = quality_report.get_score()
    logger.info(f"=== CTGAN QUALITY SCORE: {quality_score:.4f} ===")
    
    # 8. DATASET MERGE & FINAL SAVE
    balanced_df = pd.concat([df, synthetic_data], ignore_index=True)
    final_path = Path('processed_data/integrated_features_balanced_v1.csv')
    balanced_df.to_csv(final_path, index=False)
    
    logger.info(f"Balanced dataset saved to {final_path}")
    logger.info(f"New total samples: {len(balanced_df)}")
    logger.info(f"New class distribution:\n{balanced_df['target'].value_counts()}")

if __name__ == "__main__":
    main()
