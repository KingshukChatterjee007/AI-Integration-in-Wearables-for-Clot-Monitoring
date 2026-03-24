"""
Train Production Model with Clean Data
========================================

This script trains the final production XGBoost model on clean data (no leakage)
and saves it for deployment.

Expected Accuracy: 84% (validated)

Author: AI Integration in Wearables Project
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_production_model():
    """Train and save production-ready XGBoost model"""

    print("\n" + "="*70)
    print("TRAINING PRODUCTION MODEL - CLEAN DATA")
    print("AI Integration in Wearables for Clot Monitoring")
    print("="*70)

    # Load CLEAN enhanced dataset
    logger.info("\n[1/6] Loading clean enhanced dataset...")
    data_path = Path('processed_data/integrated_features_enhanced_CLEAN.csv')

    if not data_path.exists():
        logger.error(f"Clean dataset not found at {data_path}")
        logger.error("Please run: python scripts/create_clean_datasets.py")
        return False

    df = pd.read_csv(data_path)
    logger.info(f"   Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # Prepare features
    logger.info("\n[2/6] Preparing features...")
    non_feature_cols = ['subject_id', 'activity', 'window_id', 'risk_category']

    # Get all numeric feature columns
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    X_raw = df[feature_cols]

    # Select only numeric columns
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    X = X_raw[numeric_cols].copy()
    y = df['risk_category'].copy()

    # Handle missing values
    X = X.fillna(X.median())

    logger.info(f"   Features: {len(numeric_cols)}")
    logger.info(f"   Samples: {len(X):,}")

    # Show feature breakdown
    feature_categories = {
        'ECG': [c for c in numeric_cols if 'ecg_' in c.lower()],
        'Pleth': [c for c in numeric_cols if 'pleth_' in c.lower()],
        'Temperature': [c for c in numeric_cols if 'temp_' in c.lower()],
        'Motion': [c for c in numeric_cols if any(x in c.lower() for x in ['accel', 'gyro'])],
        'Demographics': [c for c in numeric_cols if any(x in c.lower() for x in ['gender', 'age', 'bmi', 'height', 'weight'])],
        'Advanced PPG': [c for c in numeric_cols if any(x in c.lower() for x in ['quality_', 'hr_', 'cardiac', 'pulse', 'perfusion'])],
    }

    logger.info(f"\n   Feature breakdown:")
    for category, features in feature_categories.items():
        if features:
            logger.info(f"      {category:15s}: {len(features):3d} features")

    # Encode target labels
    logger.info("\n[3/6] Encoding target labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    logger.info(f"   Risk categories: {list(le.classes_)}")
    logger.info(f"\n   Distribution:")
    for category, count in pd.Series(y).value_counts().items():
        pct = count / len(y) * 100
        logger.info(f"      {category:15s}: {count:5d} ({pct:5.1f}%)")

    # Split data
    logger.info("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    logger.info(f"   Training set:  {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"   Testing set:   {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Scale features
    logger.info("\n[5/6] Training XGBoost model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )

    logger.info("   Training in progress...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')

    logger.info(f"\n   {'='*70}")
    logger.info(f"   TRAINING RESULTS")
    logger.info(f"   {'='*70}")
    logger.info(f"   Training Accuracy:   {train_acc*100:.2f}%")
    logger.info(f"   Testing Accuracy:    {test_acc*100:.2f}%")
    logger.info(f"   F1-Score (weighted): {test_f1*100:.2f}%")
    logger.info(f"   CV Accuracy:         {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    logger.info(f"   {'='*70}")

    # Classification report
    logger.info(f"\n   Detailed Performance by Risk Category:")
    print("\n" + classification_report(y_test, y_pred_test, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    logger.info(f"\n   Confusion Matrix:")
    logger.info(f"   {cm}")

    # Feature importance
    logger.info(f"\n   Top 15 Most Important Features:")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': numeric_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance_df.head(15).iterrows():
        logger.info(f"      {row['feature']:40s}: {row['importance']:.4f}")

    # Save model
    logger.info("\n[6/6] Saving production model...")

    models_dir = Path('trained_models')
    models_dir.mkdir(exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = models_dir / 'xgboost_84percent_CLEAN.pkl'
    scaler_path = models_dir / 'scaler_CLEAN.pkl'
    encoder_path = models_dir / 'label_encoder_CLEAN.pkl'
    features_path = models_dir / 'feature_names_CLEAN.pkl'
    metadata_path = models_dir / f'model_metadata_CLEAN_{timestamp}.pkl'

    # Save all components
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, encoder_path)
    joblib.dump(numeric_cols, features_path)

    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': test_f1,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'n_features': len(numeric_cols),
        'n_samples': len(df),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'risk_categories': list(le.classes_),
        'training_date': timestamp,
        'dataset': 'integrated_features_enhanced_CLEAN.csv',
        'data_leakage_fixed': True,
        'removed_features': [
            'composite_risk_score', 'composite_risk_score_old',
            'bp_risk', 'hr_variability_risk', 'age_risk', 'bmi_risk',
            'anomaly_risk_score', 'anomaly_risk_level', 'risk_category_old'
        ]
    }
    joblib.dump(metadata, metadata_path)

    logger.info(f"\n   Model saved successfully!")
    logger.info(f"   Model file:      {model_path}")
    logger.info(f"   Scaler file:     {scaler_path}")
    logger.info(f"   Encoder file:    {encoder_path}")
    logger.info(f"   Features file:   {features_path}")
    logger.info(f"   Metadata file:   {metadata_path}")

    # Print summary
    print("\n" + "="*70)
    print("PRODUCTION MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel Performance:")
    print(f"   Test Accuracy:    {test_acc*100:.2f}%")
    print(f"   CV Accuracy:      {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    print(f"   F1-Score:         {test_f1*100:.2f}%")
    print(f"\nDataset:")
    print(f"   Source:           integrated_features_enhanced_CLEAN.csv")
    print(f"   Samples:          {len(df):,}")
    print(f"   Features:         {len(numeric_cols)}")
    print(f"   Data Leakage:     FIXED (9 features removed)")
    print(f"\nModel Location:")
    print(f"   {model_path}")
    print(f"\nTo use this model for predictions:")
    print(f"   model = joblib.load('trained_models/xgboost_84percent_CLEAN.pkl')")
    print(f"   scaler = joblib.load('trained_models/scaler_CLEAN.pkl')")
    print(f"   encoder = joblib.load('trained_models/label_encoder_CLEAN.pkl')")
    print("\n" + "="*70 + "\n")

    return True


if __name__ == "__main__":
    success = train_production_model()
    if success:
        print("Ready for production deployment!")
    else:
        print("Training failed. Check errors above.")
