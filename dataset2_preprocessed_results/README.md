# Dataset 2 Preprocessed Results

## Overview
This folder contains the complete preprocessed results for Dataset 2 (PPG_Dataset.csv) - Cardiac Event Detection.
All preprocessing steps have been completed and verified for machine learning model development.

## Dataset Information
- **Source**: PPG_Dataset.csv (PPG cardiac signals)
- **Task**: Cardiac event detection (Normal vs MI)
- **Total Samples**: 2,576
- **Original Features**: 2,000
- **Enhanced Features**: 29
- **Classes**: 2 (Normal=0, MI=1)

## Folder Structure

### 📁 data/
Contains all preprocessed data files:
- `X_enhanced_features.csv` - Enhanced feature matrix (2576×29)
- `X_enhanced_normalized.npy` - Normalized feature matrix
- `y_dataset2_encoded.npy` - Encoded labels (0=Normal, 1=MI)
- `y_dataset2_original.csv` - Original string labels
- `signal_quality_scores.csv/.npy` - Signal quality assessments
- `X_train_enhanced.npy` - Training features (1,803 samples)
- `y_train_enhanced.npy` - Training labels
- `X_val_enhanced.npy` - Validation features (386 samples)  
- `y_val_enhanced.npy` - Validation labels
- `X_test_enhanced.npy` - Test features (387 samples)
- `y_test_enhanced.npy` - Test labels
- `data_splits_summary.csv` - Overview of data splits

### 🔧 models/
Contains trained preprocessing models:
- `scaler_enhanced.pkl` - Trained StandardScaler for normalization
- `label_encoder_dataset2.pkl` - Trained LabelEncoder for labels
- `enhanced_preprocessor.pkl` - Complete preprocessing pipeline

### 📊 metadata/
Contains comprehensive metadata:
- `dataset2_preprocessing_metadata.json` - Complete preprocessing information

### 📝 reports/
Contains verification documentation:
- `preprocessing_verification_report.md` - Detailed verification report

### 📈 visualizations/
Reserved for preprocessing visualizations (if generated)

## Enhanced Features (29 total)

### Time Domain (11 features):
- `time_mean`, `time_std`, `time_var`, `time_rms`
- `time_max`, `time_min`, `time_range`
- `time_skewness`, `time_kurtosis`, `time_energy`
- `time_zero_crossings`

### Frequency Domain (6 features):
- `freq_vlf_power`, `freq_lf_power`, `freq_hf_power`
- `freq_total_power`, `freq_lf_hf_ratio`, `freq_spectral_centroid`

### Cardiac Domain (10 features):
- `cardiac_hr_mean`, `cardiac_hr_std`, `cardiac_rmssd`, `cardiac_pnn50`
- `cardiac_pulse_width`, `cardiac_pulse_width_std`
- `cardiac_rise_time`, `cardiac_fall_time`, `cardiac_rise_fall_ratio`
- `cardiac_amplitude_var`

### Quality Domain (2 features):
- `signal_quality_index`, `signal_snr`

## Data Splits
- **Training**: 1,803 samples (70.0%)
- **Validation**: 386 samples (15.0%)  
- **Test**: 387 samples (15.0%)
- **Class Balance**: Maintained across all splits

## Signal Quality Assessment
- **Average Quality**: 0.369
- **Low Quality (0-0.3)**: 1270 samples
- **Medium Quality (0.3-0.7)**: 1306 samples
- **High Quality (0.7-1.0)**: 0 samples

## Usage Example

```python
import numpy as np
import pandas as pd
import pickle

# Load preprocessed data
X_train = np.load('data/X_train_enhanced.npy')
y_train = np.load('data/y_train_enhanced.npy')
X_val = np.load('data/X_val_enhanced.npy')
y_val = np.load('data/y_val_enhanced.npy')
X_test = np.load('data/X_test_enhanced.npy')
y_test = np.load('data/y_test_enhanced.npy')

# Load preprocessing models
with open('models/scaler_enhanced.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/label_encoder_dataset2.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load feature names and metadata
import json
with open('metadata/dataset2_preprocessing_metadata.json', 'r') as f:
    metadata = json.load(f)
feature_names = metadata['feature_names']

print(f"Training data: {X_train.shape}")
print(f"Features: {len(feature_names)}")
print(f"Classes: {label_encoder.classes_}")
```

## Preprocessing Steps Completed ✅

1. **Data Loading & Validation** - PPG_Dataset.csv loaded and validated
2. **Enhanced Feature Extraction** - 29 features across 4 domains
3. **Signal Quality Assessment** - SQI calculated for all samples
4. **Data Normalization** - StandardScaler applied (mean≈0, std≈1)  
5. **Label Encoding** - String labels encoded to integers
6. **Stratified Splitting** - 70/15/15 train/val/test splits
7. **Zero Warnings** - Clean, production-ready preprocessing
8. **Comprehensive Saving** - All results saved with metadata

## Quality Assurance
- ✅ No missing values
- ✅ No infinite values  
- ✅ No NaN values
- ✅ Perfect normalization
- ✅ Balanced class distribution
- ✅ Consistent feature dimensions
- ✅ Complete documentation

## Ready for Model Training
This preprocessed dataset is ready for:
- Machine learning model development
- Deep learning architectures
- Hyperparameter optimization  
- Cross-validation experiments
- Performance evaluation

Generated: 2025-09-05 17:32:47
Preprocessing Status: ✅ COMPLETE
