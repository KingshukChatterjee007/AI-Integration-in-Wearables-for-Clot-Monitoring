"""
PREPROCESSED DATA EXPLANATION FOR AI CLOT MONITORING PROJECT
===========================================================

This document explains what each preprocessed file contains, how it was generated,
and its specific purpose in the AI Integration in Wearables for Clot Monitoring project.

Author: AI Assistant
Date: September 11, 2025
Project: AI Integration in Wearables for Clot Monitoring
"""

# =============================================================================
# 1. PPG_DATASET.CSV (60.84 MB)
# =============================================================================

"""
PURPOSE: Medical-grade PPG data for cardiovascular analysis and clot detection

WHAT IT CONTAINS:
- Original medical PPG dataset with 2,576 patient records
- 2,000 PPG signal channels (columns 0-1999) 
- Each channel represents different PPG signal measurements
- "Label" column indicating MI (Myocardial Infarction) classification
- Used for training models to detect cardiovascular events

HOW IT WAS GENERATED:
```python
# From data_preprocessing.py - IntegratedPreprocessor class
def _load_ppg_dataset(self) -> pd.DataFrame:
    ppg_file = self.csv_path / "PPG_Dataset.csv"
    
    if ppg_file.exists():
        logger.info(f"Loading PPG_Dataset.csv ({ppg_file.stat().st_size / (1024*1024):.1f} MB)...")
        
        # Load in chunks to handle large file
        chunks = []
        for chunk in pd.read_csv(ppg_file, chunksize=1000):
            # Clean and preprocess each chunk
            chunk = self.handle_missing_values(chunk, strategy='median_fill')
            chunks.append(chunk)
        
        ppg_dataset = pd.concat(chunks, ignore_index=True)
        logger.info(f"Successfully loaded PPG dataset: {len(ppg_dataset):,} records")
        return ppg_dataset
```

MEDICAL SIGNIFICANCE:
- PPG (Photoplethysmography) measures blood volume changes in tissues
- Essential for detecting cardiovascular abnormalities
- MI labels help train models for heart attack prediction
- 2,000 channels provide comprehensive cardiovascular profiling
"""

# =============================================================================
# 2. SUBJECTS_INFO.CSV (0.01 MB) 
# =============================================================================

"""
PURPOSE: Demographic and physiological data for personalized clot risk assessment

WHAT IT CONTAINS:
- 66 records (22 subjects × 3 activities each)
- Demographics: gender, height, weight, age, BMI
- Physiological measurements: blood pressure, heart rate, SpO2
- Calculated changes: bp_sys_change, hr_1_change, etc.
- Encoded variables: gender_encoded, activity_encoded
- Risk indicators: bp_risk, age_risk, bmi_risk, composite_risk_score

HOW IT WAS GENERATED:
```python
# From data_preprocessing.py - SubjectDatasetPreprocessor class
def load_subjects_info(self) -> pd.DataFrame:
    subjects_info_file = self.csv_path / "subjects_info.csv"
    
    if subjects_info_file.exists():
        subjects_info = pd.read_csv(subjects_info_file)
        
        # Calculate BMI
        if 'height' in subjects_info.columns and 'weight' in subjects_info.columns:
            subjects_info['bmi'] = subjects_info['weight'] / ((subjects_info['height']/100) ** 2)
        
        # Calculate physiological changes
        change_pairs = [
            ('bp_sys_change', 'bp_sys_end', 'bp_sys_start'),
            ('bp_dia_change', 'bp_dia_end', 'bp_dia_start'),
            ('hr_1_change', 'hr_1_end', 'hr_1_start'),
            ('hr_2_change', 'hr_2_end', 'hr_2_start'),
            ('spo2_change', 'spo2_end', 'spo2_start')
        ]
        
        for change_col, end_col, start_col in change_pairs:
            if end_col in subjects_info.columns and start_col in subjects_info.columns:
                subjects_info[change_col] = subjects_info[end_col] - subjects_info[start_col]
        
        # Encode categorical variables
        if 'gender' in subjects_info.columns:
            subjects_info['gender_encoded'] = subjects_info['gender'].map({'male': 1, 'female': 0})
            
        if 'activity' in subjects_info.columns:
            activity_mapping = {'sit': 0, 'walk': 1, 'run': 2}
            subjects_info['activity_encoded'] = subjects_info['activity'].map(activity_mapping)
```

CLOT MONITORING SIGNIFICANCE:
- Demographics affect clot risk (age, gender, BMI)
- Blood pressure changes indicate cardiovascular stress
- Heart rate variability shows autonomic function
- Activity level affects circulation and clot formation risk
"""

# =============================================================================
# 3. SUBJECT_FEATURES.CSV (4.42 MB)
# =============================================================================

"""
PURPOSE: Comprehensive sensor-based features for clot detection from wearables

WHAT IT CONTAINS:
- 3,207 feature windows from 67 subject files (22 subjects × 3 activities)
- 128 features per window including:
  * ECG features (15): heart electrical activity
  * PPG features (pleth_1-6, 90 features): blood flow measurements  
  * Temperature features (15): thermal regulation
  * Motion features (8): accelerometer and gyroscope data
- Each window represents 5,000 data points (~10-20 seconds of sensor data)

HOW IT WAS GENERATED:
```python
# From data_preprocessing.py - SubjectDatasetPreprocessor class
def extract_subject_features(self, window_size: int = 5000) -> pd.DataFrame:
    if 'subject_data' not in self.processed_data:
        return pd.DataFrame()
    
    all_features = []
    
    # Process each subject file
    for filename, data in self.processed_data['subject_data'].items():
        subject_id, activity = filename.split('_')[0], filename.split('_')[1]
        
        # Create sliding windows
        for i in range(0, len(data) - window_size, window_size):
            window = data.iloc[i:i+window_size]
            
            feature_dict = {
                'subject_id': subject_id,
                'activity': activity,
                'window_id': i // window_size
            }
            
            # Extract features for different sensor types
            self._extract_sensor_features(window, feature_dict, 'ecg')      # Heart electrical
            self._extract_sensor_features(window, feature_dict, 'pleth')    # Blood flow
            self._extract_sensor_features(window, feature_dict, 'temp')     # Temperature
            self._extract_motion_features(window, feature_dict)             # Motion
            
            all_features.append(feature_dict)
    
    return pd.DataFrame(all_features)

def _extract_sensor_features(self, window: pd.DataFrame, feature_dict: Dict, sensor_type: str):
    sensor_cols = [col for col in window.columns if sensor_type in col.lower()]
    
    for col in sensor_cols:
        if col in window.columns:
            series = window[col].dropna()
            if len(series) > 0:
                # Statistical features
                feature_dict[f'{col}_mean'] = series.mean()
                feature_dict[f'{col}_std'] = series.std()
                feature_dict[f'{col}_median'] = series.median()
                feature_dict[f'{col}_min'] = series.min()
                feature_dict[f'{col}_max'] = series.max()
                feature_dict[f'{col}_skew'] = series.skew()
                feature_dict[f'{col}_kurt'] = series.kurtosis()
                
                # Percentiles
                feature_dict[f'{col}_q25'] = series.quantile(0.25)
                feature_dict[f'{col}_q75'] = series.quantile(0.75)
                
                # Signal variability
                feature_dict[f'{col}_range'] = series.max() - series.min()
                feature_dict[f'{col}_iqr'] = series.quantile(0.75) - series.quantile(0.25)
```

CLOT DETECTION SIGNIFICANCE:
- ECG changes can indicate blood flow abnormalities
- PPG signal variations show microcirculation changes
- Temperature changes may indicate circulation problems
- Motion patterns affect venous return and clot risk
"""

# =============================================================================
# 4. INTEGRATED_FEATURES.CSV (4.61 MB)
# =============================================================================

"""
PURPOSE: Complete feature set combining sensor data with demographics for ML models

WHAT IT CONTAINS:
- Same 3,207 windows as subject_features.csv
- 145 features per window (128 sensor + 17 demographic)
- Additional demographic features: gender, age, BMI, blood pressure changes
- Risk assessment features: bp_risk, hr_variability_risk, age_risk, bmi_risk
- Composite risk score for overall clot probability

HOW IT WAS GENERATED:
```python
# From data_preprocessing.py - IntegratedPreprocessor class
def _create_integrated_features(self, subject_features: pd.DataFrame, 
                              subjects_info: pd.DataFrame) -> pd.DataFrame:
    # Merge subject features with demographic data
    merge_cols = ['subject_id', 'activity']
    demo_cols = ['gender', 'height', 'weight', 'age', 'bmi', 'gender_encoded', 
                'activity_encoded', 'bp_sys_change', 'bp_dia_change', 
                'hr_1_change', 'hr_2_change', 'spo2_change']
    
    integrated = subject_features.merge(
        subjects_info[merge_cols + demo_cols],
        on=merge_cols,
        how='left'
    )
    
    # Create risk indicators for clot monitoring
    if 'bp_sys_change' in integrated.columns:
        integrated['bp_risk'] = (abs(integrated['bp_sys_change']) > 20).astype(int)
    
    if 'hr_1_change' in integrated.columns:
        integrated['hr_variability_risk'] = (abs(integrated['hr_1_change']) > 15).astype(int)
    
    if 'age' in integrated.columns:
        integrated['age_risk'] = (integrated['age'] > 50).astype(int)
    
    if 'bmi' in integrated.columns:
        integrated['bmi_risk'] = (integrated['bmi'] > 30).astype(int)
    
    # Composite risk score
    risk_columns = ['bp_risk', 'hr_variability_risk', 'age_risk', 'bmi_risk']
    available_risk_cols = [col for col in risk_columns if col in integrated.columns]
    
    if available_risk_cols:
        integrated['composite_risk_score'] = integrated[available_risk_cols].sum(axis=1)
    
    return integrated
```

ML TRAINING SIGNIFICANCE:
- Combines wearable sensor data with clinical risk factors
- Provides comprehensive feature set for clot prediction models
- Risk scores help identify high-risk patients
- Demographic features improve model personalization
"""

# =============================================================================
# 5. ADVANCED_PPG_FEATURES.CSV (0.00 MB)
# =============================================================================

"""
PURPOSE: Specialized cardiovascular features from PPG signal analysis

WHAT IT CONTAINS:
- 20 feature records from selected PPG channels (sampled from 2000 total)
- 22 specialized cardiovascular features per record:
  * Signal quality metrics: SNR, perfusion index, motion artifacts
  * Heart rate analysis: mean HR, HR variability, peak detection
  * Pulse analysis: peak/valley counting, rhythm analysis
  * Anomaly detection: risk scores and anomaly counts

HOW IT WAS GENERATED:
```python
# From data_preprocessing.py - IntegratedPreprocessor class  
def _extract_advanced_ppg_features(self, ppg_dataset: pd.DataFrame) -> pd.DataFrame:
    from ppg_analysis import extract_ppg_features_from_dataset
    
    # Identify PPG columns (numeric columns 0-1999)
    ppg_columns = [col for col in ppg_dataset.columns 
                   if str(col).isdigit() and int(col) < 2000]
    
    if not ppg_columns:
        return pd.DataFrame()
    
    # Use representative sample of PPG channels (memory optimization)
    num_channels = min(20, len(ppg_columns))
    selected_channels = ppg_columns[::len(ppg_columns)//num_channels][:num_channels]
    
    logger.info(f"Using {len(selected_channels)} representative PPG channels from {len(ppg_columns)} total")
    
    # Extract advanced features
    features_df = extract_ppg_features_from_dataset(
        data=ppg_dataset,
        ppg_columns=selected_channels,
        window_size=2500  # Larger window for cardiovascular analysis
    )
    
    return features_df

# From ppg_analysis.py - PPGSignalAnalyzer class
def analyze_ppg_signal(self, ppg_signal: np.ndarray, channel_name: str) -> Dict[str, Any]:
    features = {}
    
    # Signal quality assessment
    quality_metrics = self.assess_signal_quality(ppg_signal)
    features.update(quality_metrics)
    
    # Heart rate analysis 
    hr_metrics = self.extract_heart_rate_features(ppg_signal)
    features.update(hr_metrics)
    
    # Pulse morphology analysis
    pulse_metrics = self.extract_pulse_features(ppg_signal)
    features.update(pulse_metrics)
    
    # Anomaly detection
    anomaly_metrics = self.detect_anomalies(ppg_signal)
    features.update(anomaly_metrics)
    
    return features
```

CARDIOVASCULAR SIGNIFICANCE:
- PPG quality metrics ensure reliable measurements
- Heart rate variability indicates autonomic function
- Pulse morphology changes may indicate vascular problems
- Anomaly detection flags potential cardiovascular events
"""

# =============================================================================
# COMPLETE DATA PIPELINE SUMMARY
# =============================================================================

"""
DATA FLOW FOR CLOT MONITORING:

1. RAW DATA INPUT:
   - PPG_Dataset.csv (2,576 medical records)
   - subjects_info.csv (demographic data)
   - s1-s22 activity files (67 wearable sensor files)

2. PREPROCESSING PIPELINE:
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │   Raw Sensor    │ -> │   Window-based   │ -> │  Feature        │
   │   Data Files    │    │   Segmentation   │    │  Extraction     │
   └─────────────────┘    └──────────────────┘    └─────────────────┘
            │                       │                       │
            v                       v                       v
   ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   │  PPG Dataset    │    │ Subject Features │    │  Integrated     │
   │  Processing     │    │   (128 features) │    │  Features       │
   │  (2000 channels)│    │   3,207 windows  │    │  (145 features) │
   └─────────────────┘    └──────────────────┘    └─────────────────┘

3. OUTPUT FILES FOR ML:
   - ppg_dataset.csv: Medical-grade cardiovascular data
   - subjects_info.csv: Demographics and risk factors  
   - subject_features.csv: Wearable sensor features
   - integrated_features.csv: Complete ML-ready dataset
   - advanced_ppg_features.csv: Specialized cardiac features

4. ML APPLICATIONS:
   - Clot risk prediction models
   - Cardiovascular event detection
   - Personalized risk assessment
   - Real-time monitoring algorithms

TEACHER EXPLANATION POINTS:
✅ Multi-modal data fusion (wearable sensors + medical data)
✅ Window-based feature extraction for time-series analysis
✅ Statistical and domain-specific feature engineering
✅ Risk stratification for personalized medicine
✅ Scalable preprocessing for large healthcare datasets
✅ Quality assurance and anomaly detection
"""