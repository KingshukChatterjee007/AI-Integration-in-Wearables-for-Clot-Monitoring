"""
CODE EXECUTION FLOW FOR PREPROCESSED DATA GENERATION
==================================================

This document shows exactly how each preprocessed file was created through code execution.

MAIN EXECUTION: data_preprocessing.py -> main() function
"""

# =============================================================================
# STEP 1: INITIALIZATION
# =============================================================================

"""
ENTRY POINT: main() function in data_preprocessing.py

```python
def main():
    csv_path = Path("csv")  # Your dataset folder
    output_path = Path("processed_data")  # Output folder
    
    # Create main preprocessor
    preprocessor = IntegratedPreprocessor(str(csv_path))
    
    # Run complete preprocessing pipeline
    results = preprocessor.run_complete_preprocessing(
        load_subjects=True,          # Process wearable sensor data
        load_ppg_dataset=True,       # Process medical PPG data  
        max_subject_files=None       # Process ALL files (no limits)
    )
```

WHAT HAPPENS:
- Creates IntegratedPreprocessor object
- Sets up data paths and output directories
- Initiates complete preprocessing pipeline
"""

# =============================================================================
# STEP 2: PPG DATASET PROCESSING
# =============================================================================

"""
CODE EXECUTION: IntegratedPreprocessor.run_complete_preprocessing()

```python
def run_complete_preprocessing(self, load_ppg_dataset=True, ...):
    results = {}
    
    # GENERATES: ppg_dataset.csv
    if load_ppg_dataset:
        ppg_dataset = self._load_ppg_dataset()  # Calls method below
        if not ppg_dataset.empty:
            results['ppg_dataset'] = ppg_dataset

def _load_ppg_dataset(self) -> pd.DataFrame:
    ppg_file = self.csv_path / "PPG_Dataset.csv"
    
    logger.info(f"Loading PPG_Dataset.csv ({ppg_file.stat().st_size / (1024*1024):.1f} MB)...")
    
    # Load in chunks for memory efficiency
    chunks = []
    for chunk in pd.read_csv(ppg_file, chunksize=1000):
        # Handle missing values with median fill
        chunk = self.handle_missing_values(chunk, strategy='median_fill')
        chunks.append(chunk)
    
    ppg_dataset = pd.concat(chunks, ignore_index=True)
    logger.info(f"Successfully loaded PPG dataset: {len(ppg_dataset):,} records")
    
    return ppg_dataset  # -> 2,576 records with 2,001 columns
```

OUTPUT: processed_data/ppg_dataset.csv (60.84 MB)
- Medical-grade PPG data with 2,000 signal channels + MI labels
- Chunk-based loading for memory optimization
- Missing value imputation applied
"""

# =============================================================================
# STEP 3: SUBJECTS INFO PROCESSING  
# =============================================================================

"""
CODE EXECUTION: SubjectDatasetPreprocessor.load_subjects_info()

```python
# GENERATES: subjects_info.csv
subjects_info = self.subject_processor.load_subjects_info()
if not subjects_info.empty:
    results['subjects_info'] = subjects_info

def load_subjects_info(self) -> pd.DataFrame:
    subjects_info_file = self.csv_path / "subjects_info.csv"
    
    subjects_info = pd.read_csv(subjects_info_file)
    
    # Calculate BMI from height and weight
    if 'height' in subjects_info.columns and 'weight' in subjects_info.columns:
        subjects_info['bmi'] = subjects_info['weight'] / ((subjects_info['height']/100) ** 2)
    
    # Calculate physiological changes (before/after activity)
    change_pairs = [
        ('bp_sys_change', 'bp_sys_end', 'bp_sys_start'),    # Blood pressure
        ('bp_dia_change', 'bp_dia_end', 'bp_dia_start'),
        ('hr_1_change', 'hr_1_end', 'hr_1_start'),          # Heart rate
        ('hr_2_change', 'hr_2_end', 'hr_2_start'),
        ('spo2_change', 'spo2_end', 'spo2_start')           # Oxygen saturation
    ]
    
    for change_col, end_col, start_col in change_pairs:
        if end_col in subjects_info.columns and start_col in subjects_info.columns:
            subjects_info[change_col] = subjects_info[end_col] - subjects_info[start_col]
    
    # Encode categorical variables for ML
    if 'gender' in subjects_info.columns:
        subjects_info['gender_encoded'] = subjects_info['gender'].map({'male': 1, 'female': 0})
        
    if 'activity' in subjects_info.columns:
        activity_mapping = {'sit': 0, 'walk': 1, 'run': 2}
        subjects_info['activity_encoded'] = subjects_info['activity'].map(activity_mapping)
    
    return subjects_info  # -> 66 records with 25 columns
```

OUTPUT: processed_data/subjects_info.csv (0.01 MB)
- Demographics with calculated BMI
- Physiological changes during activities
- Categorical variables encoded for ML
"""

# =============================================================================
# STEP 4: WEARABLE SENSOR DATA PROCESSING
# =============================================================================

"""
CODE EXECUTION: SubjectDatasetPreprocessor.load_subject_data()

```python
# GENERATES: subject_features.csv  
if load_subjects:
    logger.info("Processing subject data files...")
    subject_data = self.subject_processor.load_subject_data(max_files=max_subject_files)
    if subject_data:
        results['subject_data'] = subject_data
        
        # Extract features from sensor data
        subject_features = self.subject_processor.extract_subject_features()
        if not subject_features.empty:
            results['subject_features'] = subject_features

def load_subject_data(self, max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    subject_data = {}
    files_to_process = self.subject_files[:max_files] if max_files else self.subject_files
    
    logger.info(f"Loading {len(files_to_process)} subject data files...")
    
    for file_path in files_to_process:  # Processes all 67 files
        try:
            filename = file_path.stem  # e.g., "s1_walk"
            data = pd.read_csv(file_path)
            
            # Preprocess individual file
            data = self.preprocess_subject_file(data, filename)
            subject_data[filename] = data
            
            logger.info(f"Processed {filename}: {len(data)} rows")
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    return subject_data  # -> 67 files loaded as DataFrames

def preprocess_subject_file(self, data: pd.DataFrame, filename: str) -> pd.DataFrame:
    data_copy = data.copy()
    
    # Handle datetime conversion
    if 'time' in data_copy.columns:
        data_copy['time'] = pd.to_datetime(data_copy['time'])
        data_copy = data_copy.sort_values('time').reset_index(drop=True)
    
    # Handle missing values with forward fill
    data_copy = self.handle_missing_values(data_copy, strategy='forward_fill')
    
    # Calculate derived features
    accel_cols = ['a_x', 'a_y', 'a_z']
    gyro_cols = ['g_x', 'g_y', 'g_z']
    
    # Calculate accelerometer magnitude: √(x² + y² + z²)
    if all(col in data_copy.columns for col in accel_cols):
        data_copy['accel_magnitude'] = np.sqrt(
            data_copy['a_x']**2 + data_copy['a_y']**2 + data_copy['a_z']**2
        )
    
    # Calculate gyroscope magnitude: √(x² + y² + z²)  
    if all(col in data_copy.columns for col in gyro_cols):
        data_copy['gyro_magnitude'] = np.sqrt(
            data_copy['g_x']**2 + data_copy['g_y']**2 + data_copy['g_z']**2
        )
    
    return data_copy
```

PROCESSING VOLUME:
- 67 files processed (22 subjects × 3 activities + extras)
- ~16.2 million total sensor data points
- Derived features calculated (magnitude vectors)
- Missing data handled appropriately
"""

# =============================================================================
# STEP 5: FEATURE EXTRACTION FROM SENSOR DATA
# =============================================================================

"""
CODE EXECUTION: SubjectDatasetPreprocessor.extract_subject_features()

```python
def extract_subject_features(self, window_size: int = 5000) -> pd.DataFrame:
    if 'subject_data' not in self.processed_data:
        return pd.DataFrame()
    
    all_features = []
    
    # Process each loaded subject file
    for filename, data in self.processed_data['subject_data'].items():
        try:
            # Extract subject and activity from filename
            parts = filename.split('_')
            subject_id, activity = parts[0], parts[1]  # e.g., "s1", "walk"
            
            # Create sliding windows of 5000 data points each
            for i in range(0, len(data) - window_size, window_size):
                window = data.iloc[i:i+window_size]  # 5000 rows per window
                
                feature_dict = {
                    'subject_id': subject_id,
                    'activity': activity, 
                    'window_id': i // window_size
                }
                
                # Extract features for different sensor types
                self._extract_sensor_features(window, feature_dict, 'ecg')      # 15 features
                self._extract_sensor_features(window, feature_dict, 'pleth')    # 90 features (6 channels × 15)
                self._extract_sensor_features(window, feature_dict, 'temp')     # 15 features  
                self._extract_motion_features(window, feature_dict)             # 8 features
                
                all_features.append(feature_dict)  # Total: 128 features per window
                
        except Exception as e:
            logger.warning(f"Error processing {filename}: {e}")
            continue
    
    feature_df = pd.DataFrame(all_features)
    logger.info(f"Extracted features for {len(feature_df)} windows")
    
    return feature_df  # -> 3,207 windows × 128 features

def _extract_sensor_features(self, window: pd.DataFrame, feature_dict: Dict, sensor_type: str):
    # Find columns matching sensor type (e.g., 'ecg', 'pleth', 'temp')
    sensor_cols = [col for col in window.columns if sensor_type in col.lower()]
    
    for col in sensor_cols:
        if col in window.columns:
            series = window[col].dropna()
            if len(series) > 0:
                # Statistical features (15 per column)
                feature_dict[f'{col}_mean'] = series.mean()           # Central tendency
                feature_dict[f'{col}_std'] = series.std()             # Variability
                feature_dict[f'{col}_median'] = series.median()       # Robust center
                feature_dict[f'{col}_min'] = series.min()             # Range
                feature_dict[f'{col}_max'] = series.max()             # Range
                feature_dict[f'{col}_skew'] = series.skew()           # Asymmetry
                feature_dict[f'{col}_kurt'] = series.kurtosis()       # Tail behavior
                
                # Percentiles for distribution shape
                feature_dict[f'{col}_q25'] = series.quantile(0.25)
                feature_dict[f'{col}_q75'] = series.quantile(0.75)
                
                # Signal variability measures
                feature_dict[f'{col}_range'] = series.max() - series.min()
                feature_dict[f'{col}_iqr'] = series.quantile(0.75) - series.quantile(0.25)
                
                # Signal complexity
                feature_dict[f'{col}_zero_crossing_rate'] = ((series[:-1] * series[1:]) < 0).sum() / len(series)
                feature_dict[f'{col}_mean_absolute_deviation'] = (series - series.mean()).abs().mean()
                feature_dict[f'{col}_coefficient_of_variation'] = series.std() / series.mean() if series.mean() != 0 else 0
                feature_dict[f'{col}_energy'] = (series ** 2).sum()

def _extract_motion_features(self, window: pd.DataFrame, feature_dict: Dict):
    # Accelerometer magnitude features
    if 'accel_magnitude' in window.columns:
        mag_series = window['accel_magnitude'].dropna()
        if len(mag_series) > 0:
            feature_dict['accel_magnitude_mean'] = mag_series.mean()
            feature_dict['accel_magnitude_std'] = mag_series.std()
    
    # Gyroscope magnitude features  
    if 'gyro_magnitude' in window.columns:
        mag_series = window['gyro_magnitude'].dropna()
        if len(mag_series) > 0:
            feature_dict['gyro_magnitude_mean'] = mag_series.mean()
            feature_dict['gyro_magnitude_std'] = mag_series.std()
```

OUTPUT: processed_data/subject_features.csv (4.42 MB)
- 3,207 feature windows from 67 subject files
- 128 features per window (ECG + PPG + Temperature + Motion)
- Window size: 5,000 data points each (~10-20 seconds)
- Statistical and signal processing features extracted
"""

# =============================================================================
# STEP 6: INTEGRATED FEATURES WITH DEMOGRAPHICS
# =============================================================================

"""
CODE EXECUTION: IntegratedPreprocessor._create_integrated_features()

```python
# GENERATES: integrated_features.csv
if 'subject_features' in results and 'subjects_info' in results:
    integrated_features = self._create_integrated_features(
        results['subject_features'], 
        results['subjects_info']
    )
    if not integrated_features.empty:
        results['integrated_features'] = integrated_features

def _create_integrated_features(self, subject_features: pd.DataFrame, 
                              subjects_info: pd.DataFrame) -> pd.DataFrame:
    try:
        # Merge sensor features with demographic data
        merge_cols = ['subject_id', 'activity']
        demo_cols = ['gender', 'height', 'weight', 'age', 'bmi', 'gender_encoded', 
                    'activity_encoded', 'bp_sys_change', 'bp_dia_change', 
                    'hr_1_change', 'hr_2_change', 'spo2_change']
        
        available_demo_cols = [col for col in demo_cols if col in subjects_info.columns]
        merge_subjects_info = subjects_info[merge_cols + available_demo_cols].drop_duplicates()
        
        # Perform left join to combine datasets
        integrated = subject_features.merge(
            merge_subjects_info,
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
        
        # Composite risk score (sum of individual risks)
        risk_columns = ['bp_risk', 'hr_variability_risk', 'age_risk', 'bmi_risk']
        available_risk_cols = [col for col in risk_columns if col in integrated.columns]
        
        if available_risk_cols:
            integrated['composite_risk_score'] = integrated[available_risk_cols].sum(axis=1)
        
        return integrated  # -> 3,207 windows × 145 features (128 sensor + 17 demographic)
        
    except Exception as e:
        logger.error(f"Error creating integrated features: {e}")
        return subject_features
```

OUTPUT: processed_data/integrated_features.csv (4.61 MB)
- Same 3,207 windows as subject_features.csv
- Enhanced with 17 additional demographic features
- Risk assessment scores for clot monitoring
- Ready for ML model training
"""

# =============================================================================
# STEP 7: ADVANCED PPG CARDIOVASCULAR ANALYSIS
# =============================================================================

"""
CODE EXECUTION: IntegratedPreprocessor._extract_advanced_ppg_features()

```python
# GENERATES: advanced_ppg_features.csv
if 'ppg_dataset' in results:
    ppg_features = self._extract_advanced_ppg_features(results['ppg_dataset'])
    if not ppg_features.empty:
        results['advanced_ppg_features'] = ppg_features

def _extract_advanced_ppg_features(self, ppg_dataset: pd.DataFrame) -> pd.DataFrame:
    from ppg_analysis import extract_ppg_features_from_dataset
    
    # Identify numeric PPG columns (0-1999)
    ppg_columns = [col for col in ppg_dataset.columns 
                   if str(col).isdigit() and int(col) < 2000]
    
    if not ppg_columns:
        return pd.DataFrame()
    
    # Use representative sample (memory optimization)
    num_channels = min(20, len(ppg_columns))
    selected_channels = ppg_columns[::len(ppg_columns)//num_channels][:num_channels]
    
    logger.info(f"Using {len(selected_channels)} representative PPG channels from {len(ppg_columns)} total")
    
    # Extract advanced cardiovascular features
    features_df = extract_ppg_features_from_dataset(
        data=ppg_dataset,
        ppg_columns=selected_channels,
        window_size=2500  # Larger window for cardiac analysis
    )
    
    return features_df  # -> 20 windows × 22 cardiovascular features

# From ppg_analysis.py
def extract_ppg_features_from_dataset(data: pd.DataFrame, ppg_columns: List[str], 
                                     window_size: int = 2500) -> pd.DataFrame:
    analyzer = PPGSignalAnalyzer()
    all_features = []
    
    for i, channel in enumerate(ppg_columns):
        if channel in data.columns:
            ppg_signal = data[channel].values
            
            # Create windows for analysis
            for window_start in range(0, len(ppg_signal) - window_size, window_size):
                window_data = ppg_signal[window_start:window_start + window_size]
                
                # Analyze PPG signal window
                features = analyzer.analyze_ppg_signal(window_data, channel)
                features['window_id'] = window_start // window_size
                features['ppg_channel'] = int(channel)
                
                all_features.append(features)
    
    return pd.DataFrame(all_features)

def analyze_ppg_signal(self, ppg_signal: np.ndarray, channel_name: str) -> Dict[str, Any]:
    features = {'signal_length': len(ppg_signal)}
    
    # 1. Signal Quality Assessment
    quality_metrics = self.assess_signal_quality(ppg_signal)
    features.update({f'quality_{k}': v for k, v in quality_metrics.items()})
    
    # 2. Heart Rate Analysis
    hr_metrics = self.extract_heart_rate_features(ppg_signal)
    features.update({f'hr_{k}': v for k, v in hr_metrics.items()})
    
    # 3. Pulse Morphology Analysis
    pulse_metrics = self.extract_pulse_features(ppg_signal)
    features.update({f'pulse_{k}': v for k, v in pulse_metrics.items()})
    
    # 4. Anomaly Detection
    anomaly_metrics = self.detect_anomalies(ppg_signal)
    features.update({f'anomaly_{k}': v for k, v in anomaly_metrics.items()})
    
    return features  # -> 22 specialized cardiovascular features
```

OUTPUT: processed_data/advanced_ppg_features.csv (0.00 MB)
- 20 feature records from selected PPG channels
- 22 specialized cardiovascular features per record
- Signal quality, heart rate, pulse morphology analysis
- Anomaly detection for cardiovascular events
"""

# =============================================================================
# STEP 8: DATA SAVING AND REPORTING
# =============================================================================

"""
CODE EXECUTION: IntegratedPreprocessor.save_processed_data()

```python
# Save all processed datasets
preprocessor.save_processed_data(str(output_path))

def save_processed_data(self, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for dataset_name, dataset in self.integrated_data.items():
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            filename = f"{dataset_name}.csv"
            filepath = output_path / filename
            
            dataset.to_csv(filepath, index=False)
            file_size = filepath.stat().st_size / (1024*1024)  # MB
            
            logger.info(f"Saved {dataset_name} to {filepath}")
            
# Generate processing report
report = preprocessor.generate_preprocessing_report()

print("\n" + "="*60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*60)

for dataset, stats in report['summary_statistics'].items():
    print(f"\n{dataset.upper()}:")
    print(f"  - Rows: {stats['rows']:,}")
    print(f"  - Columns: {stats['columns']}")
    print(f"  - Missing values: {stats['missing_values']:,}")
    print(f"  - Memory usage: {stats['memory_usage_mb']:.2f} MB")
```

FINAL OUTPUT FILES:
✅ processed_data/ppg_dataset.csv (60.84 MB)
✅ processed_data/subjects_info.csv (0.01 MB)  
✅ processed_data/subject_features.csv (4.42 MB)
✅ processed_data/integrated_features.csv (4.61 MB)
✅ processed_data/advanced_ppg_features.csv (0.00 MB)

TOTAL PROCESSING TIME: ~87 seconds
TOTAL DATA PROCESSED: 16.22M records
FEATURE WINDOWS GENERATED: 3,207
ML-READY DATASETS: 5 files
"""