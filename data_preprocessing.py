"""
Data Preprocessing for AI Integration in Wearables for Clot Monitoring
========================================================================

Simplified, working version of the preprocessing pipeline.

Author: AI Assistant
Project: AI Integration in Wearables for Clot Monitoring
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def ensure_directory_exists(path_str):
    """Verify that the directory exists and is actually a directory"""
    path = Path(path_str)
    if not path.exists():
        logger.error(f"Path does not exist: {path}")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
        return False
    if not path.is_dir():
        logger.error(f"Path is not a directory: {path}")
        return False
    return True


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert values to float with fallback"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


class DataPreprocessor:
    """Base class for data preprocessing operations"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.processed_data = {}
        self.metadata = {}
        self.scalers = {}
        
    def detect_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, pd.Index]:
        """Detect outliers using IQR or Z-score method"""
        outliers = {}
        
        for col in columns:
            if col not in data.columns:
                continue
                
            try:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers[col] = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                elif method == 'zscore':
                    z_scores = np.abs(zscore(data[col].dropna()))
                    outliers[col] = data[np.abs(zscore(data[col].dropna())) > 3].index
            except Exception as e:
                logger.warning(f"Could not detect outliers for {col}: {e}")
                outliers[col] = pd.Index([])
                
        return outliers
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values using various strategies"""
        data_copy = data.copy()
        
        try:
            if strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            elif strategy == 'forward_fill':
                return data_copy.ffill().bfill()
            else:
                return data_copy
            
            numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_copy[numeric_cols] = imputer.fit_transform(data_copy[numeric_cols])
                
        except Exception as e:
            logger.warning(f"Error handling missing values: {e}")
            
        return data_copy
    
    def filter_signal(self, signal_data: pd.Series, filter_type: str = 'butterworth', 
                     lowcut: float = 0.5, highcut: float = 40, fs: float = 500) -> pd.Series:
        """Apply digital filters to signal data"""
        try:
            # Convert pandas Series to numpy array explicitly
            signal_array = np.asarray(signal_data.values)
            
            if filter_type == 'butterworth':
                # Bandpass Butterworth filter
                nyquist = 0.5 * fs
                low = lowcut / nyquist
                high = highcut / nyquist
                
                # Ensure frequencies are valid
                if low <= 0 or high >= 1 or low >= high:
                    logger.warning("Invalid filter frequencies, returning original signal")
                    return signal_data
                
                # Fix: Use robust filter design with proper error handling
                try:
                    # Use simple lowpass filter as more stable alternative
                    if high > 0.4:  # Avoid high frequency issues
                        high = 0.4
                    if low < 0.01:  # Avoid very low frequencies
                        low = 0.01
                    
                    # Create Butterworth filter with error handling
                    sos = signal.butter(4, [low, high], btype='band', output='sos')
                    filtered_signal = signal.sosfilt(sos, signal_array)
                    
                except Exception as filter_error:
                    logger.warning(f"Butterworth filter failed: {filter_error}, using median filter instead")
                    filtered_signal = signal.medfilt(signal_array, kernel_size=5)
                
            elif filter_type == 'median':
                # Median filter for noise removal
                filtered_signal = signal.medfilt(signal_array, kernel_size=5)
            else:
                filtered_signal = signal_array
                
            return pd.Series(filtered_signal, index=signal_data.index)
            
        except Exception as e:
            logger.warning(f"Signal filtering failed: {e}. Returning original signal.")
            return signal_data
    
    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, object]:
        """Normalize data using different scaling methods"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Method must be 'standard', 'robust', or 'minmax'")
        
        data_scaled = data.copy()
        if len(numeric_cols) > 0:
            data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        return data_scaled, scaler


class SubjectDatasetPreprocessor(DataPreprocessor):
    """Specialized preprocessor for subject datasets (s1-s22 with activities)"""
    
    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.subject_files = list(self.data_path.glob('s*_*.csv'))
        self.subjects_info_file = self.data_path / 'subjects_info.csv'
        
    def load_subjects_info(self) -> pd.DataFrame:
        """Load and preprocess subjects demographic information"""
        try:
            if not self.subjects_info_file.exists():
                logger.error(f"Subjects info file not found: {self.subjects_info_file}")
                return pd.DataFrame()
                
            subjects_info = pd.read_csv(self.subjects_info_file)
            
            # Clean column names
            subjects_info.columns = subjects_info.columns.str.strip()
            
            # Extract subject ID and activity from record column
            if 'record' in subjects_info.columns:
                subjects_info[['subject_id', 'activity']] = subjects_info['record'].str.extract(r'(s\d+)_(\w+)')
            
            # Convert data types
            numeric_cols = ['height', 'weight', 'age', 'bp_sys_start', 'bp_sys_end', 
                          'bp_dia_start', 'bp_dia_end', 'hr_1_start', 'hr_1_end', 
                          'hr_2_start', 'hr_2_end', 'spo2_start', 'spo2_end']
            
            for col in numeric_cols:
                if col in subjects_info.columns:
                    subjects_info[col] = pd.to_numeric(subjects_info[col], errors='coerce')
            
            # Calculate derived features
            if all(col in subjects_info.columns for col in ['weight', 'height']):
                subjects_info['bmi'] = subjects_info['weight'] / (subjects_info['height'] / 100) ** 2
                
            # Calculate changes
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
            
            self.processed_data['subjects_info'] = subjects_info
            logger.info(f"Loaded subjects info: {len(subjects_info)} records")
            
            return subjects_info
            
        except Exception as e:
            logger.error(f"Error loading subjects info: {e}")
            return pd.DataFrame()
    
    def load_subject_data(self, max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load and preprocess individual subject data files"""
        subject_data = {}
        files_to_process = self.subject_files[:max_files] if max_files else self.subject_files
        
        logger.info(f"Loading {len(files_to_process)} subject data files...")
        
        for file_path in files_to_process:
            try:
                filename = file_path.stem
                
                # Load data with error handling
                data = pd.read_csv(file_path)
                
                # Basic preprocessing
                data = self.preprocess_subject_file(data, filename)
                subject_data[filename] = data
                
                logger.info(f"Processed {filename}: {len(data)} rows")
                
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        self.processed_data['subject_data'] = subject_data
        return subject_data
    
    def preprocess_subject_file(self, data: pd.DataFrame, filename: str) -> pd.DataFrame:
        """Preprocess individual subject data file"""
        data_copy = data.copy()
        
        # Handle datetime conversion
        if 'time' in data_copy.columns:
            try:
                data_copy['time'] = pd.to_datetime(data_copy['time'])
                data_copy = data_copy.sort_values('time').reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not convert time column for {filename}: {e}")
        
        # Handle missing values
        data_copy = self.handle_missing_values(data_copy, strategy='forward_fill')
        
        # Calculate derived features
        accel_cols = ['a_x', 'a_y', 'a_z']
        gyro_cols = ['g_x', 'g_y', 'g_z']
        
        # Accelerometer magnitude
        if all(col in data_copy.columns for col in accel_cols):
            try:
                data_copy['accel_magnitude'] = np.sqrt(
                    data_copy['a_x']**2 + data_copy['a_y']**2 + data_copy['a_z']**2
                )
            except Exception as e:
                logger.warning(f"Could not calculate accelerometer magnitude: {e}")
        
        # Gyroscope magnitude
        if all(col in data_copy.columns for col in gyro_cols):
            try:
                data_copy['gyro_magnitude'] = np.sqrt(
                    data_copy['g_x']**2 + data_copy['g_y']**2 + data_copy['g_z']**2
                )
            except Exception as e:
                logger.warning(f"Could not calculate gyroscope magnitude: {e}")
        
        return data_copy
    
    def extract_subject_features(self, window_size: int = 5000) -> pd.DataFrame:
        """Extract features from subject data using sliding windows"""
        if 'subject_data' not in self.processed_data:
            logger.error("Subject data not loaded. Call load_subject_data() first.")
            return pd.DataFrame()
        
        all_features = []
        
        for filename, data in self.processed_data['subject_data'].items():
            try:
                # Extract subject and activity from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    subject_id, activity = parts[0], parts[1]
                else:
                    subject_id, activity = filename, 'unknown'
                
                # Window-based feature extraction
                for i in range(0, len(data) - window_size, window_size):
                    window = data.iloc[i:i+window_size]
                    
                    feature_dict = {
                        'subject_id': subject_id,
                        'activity': activity,
                        'window_id': i // window_size
                    }
                    
                    # Extract features for different sensor types
                    self._extract_sensor_features(window, feature_dict, 'ecg')
                    self._extract_sensor_features(window, feature_dict, 'pleth')
                    self._extract_sensor_features(window, feature_dict, 'temp')
                    self._extract_motion_features(window, feature_dict)
                    
                    all_features.append(feature_dict)
                    
            except Exception as e:
                logger.warning(f"Error processing {filename}: {e}")
                continue
        
        if not all_features:
            logger.warning("No features extracted")
            return pd.DataFrame()
        
        feature_df = pd.DataFrame(all_features)
        self.processed_data['subject_features'] = feature_df
        
        logger.info(f"Extracted features for {len(feature_df)} windows")
        return feature_df
    
    def _extract_sensor_features(self, window: pd.DataFrame, feature_dict: Dict[str, Any], sensor_type: str) -> None:
        """Extract statistical features for a specific sensor type"""
        sensor_cols = [col for col in window.columns if sensor_type in col.lower()]
        
        for col in sensor_cols:
            try:
                # Get the data and explicitly convert to numpy array
                values = np.asarray(window[col].dropna().values)
                
                if len(values) == 0:
                    continue
                
                # Basic statistical features using pandas methods (more robust)
                series = pd.Series(values)
                
                feature_dict[f'{col}_mean'] = safe_float_conversion(series.mean())
                feature_dict[f'{col}_std'] = safe_float_conversion(series.std())
                feature_dict[f'{col}_median'] = safe_float_conversion(series.median())
                feature_dict[f'{col}_min'] = safe_float_conversion(series.min())
                feature_dict[f'{col}_max'] = safe_float_conversion(series.max())
                feature_dict[f'{col}_skew'] = safe_float_conversion(series.skew())
                feature_dict[f'{col}_kurt'] = safe_float_conversion(series.kurtosis())
                
                # Percentiles
                feature_dict[f'{col}_q25'] = safe_float_conversion(series.quantile(0.25))
                feature_dict[f'{col}_q75'] = safe_float_conversion(series.quantile(0.75))
                
                # Signal variability
                feature_dict[f'{col}_range'] = safe_float_conversion(series.max() - series.min())
                feature_dict[f'{col}_iqr'] = safe_float_conversion(series.quantile(0.75) - series.quantile(0.25))
                
                # Zero crossing rate (simplified)
                try:
                    mean_val = np.mean(values)  # Use proper numpy array
                    centered = values - mean_val
                    zero_crossings = np.where(np.diff(np.signbit(centered)))[0]
                    feature_dict[f'{col}_zero_crossing_rate'] = safe_float_conversion(len(zero_crossings) / len(values))
                except:
                    feature_dict[f'{col}_zero_crossing_rate'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Error extracting features for {col}: {e}")
                continue
    
    def _extract_motion_features(self, window: pd.DataFrame, feature_dict: Dict[str, Any]) -> None:
        """Extract motion-specific features from accelerometer and gyroscope data"""
        
        # Accelerometer features
        if 'accel_magnitude' in window.columns:
            try:
                mag_series = window['accel_magnitude'].dropna()
                if len(mag_series) > 0:
                    feature_dict['accel_magnitude_mean'] = safe_float_conversion(mag_series.mean())
                    feature_dict['accel_magnitude_std'] = safe_float_conversion(mag_series.std())
                    
                    # Activity level (number of high activity points)
                    threshold = mag_series.mean() + 2 * mag_series.std()
                    feature_dict['accel_activity_level'] = safe_float_conversion(
                        (mag_series > threshold).sum()
                    )
            except Exception as e:
                logger.debug(f"Error extracting accelerometer features: {e}")
        
        # Gyroscope features
        if 'gyro_magnitude' in window.columns:
            try:
                mag_series = window['gyro_magnitude'].dropna()
                if len(mag_series) > 0:
                    feature_dict['gyro_magnitude_mean'] = safe_float_conversion(mag_series.mean())
                    feature_dict['gyro_magnitude_std'] = safe_float_conversion(mag_series.std())
            except Exception as e:
                logger.debug(f"Error extracting gyroscope features: {e}")


class IntegratedPreprocessor:
    """Main preprocessor that integrates all datasets"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        # Add this check to prevent the directory-as-file error:
        if not ensure_directory_exists(csv_path):
            logger.error(f"CSV path is invalid: {csv_path}")
            # Create a fallback directory if needed
            self.csv_path = Path("./temp_csv")
            self.csv_path.mkdir(exist_ok=True)
        
        self.subject_processor = SubjectDatasetPreprocessor(str(self.csv_path))
        self.integrated_data = {}
        
    def run_complete_preprocessing(self, 
                                 load_subjects: bool = True,
                                 load_ppg_dataset: bool = False,
                                 max_subject_files: Optional[int] = 10) -> Dict[str, Any]:
        """Run complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline...")
        
        results = {}
        
        # Process large PPG dataset if requested
        if load_ppg_dataset:
            ppg_dataset = self._load_ppg_dataset()
            if not ppg_dataset.empty:
                results['ppg_dataset'] = ppg_dataset
                logger.info(f"Loaded PPG dataset: {len(ppg_dataset)} records")
        
        # Process subjects info (always load this as it's small)
        subjects_info = self.subject_processor.load_subjects_info()
        if not subjects_info.empty:
            results['subjects_info'] = subjects_info
        
        # Process subject data files
        if load_subjects:
            logger.info("Processing subject data files...")
            subject_data = self.subject_processor.load_subject_data(max_files=max_subject_files)
            if subject_data:
                results['subject_data'] = subject_data
                
                # Extract features
                subject_features = self.subject_processor.extract_subject_features()
                if not subject_features.empty:
                    results['subject_features'] = subject_features
        
        # Create integrated features
        if 'subjects_info' in results and 'subject_features' in results:
            integrated_features = self._create_integrated_features(
                results['subject_features'], results['subjects_info']
            )
            if not integrated_features.empty:
                results['integrated_features'] = integrated_features
        
        # Extract advanced PPG features if PPG dataset was loaded
        if 'ppg_dataset' in results:
            ppg_features = self._extract_advanced_ppg_features(results['ppg_dataset'])
            if not ppg_features.empty:
                results['advanced_ppg_features'] = ppg_features
                logger.info(f"Extracted advanced PPG features: {len(ppg_features)} windows")
        
        self.integrated_data = results
        logger.info("Complete preprocessing pipeline finished!")
        
        return results
    
    def _create_integrated_features(self, subject_features: pd.DataFrame, 
                                  subjects_info: pd.DataFrame) -> pd.DataFrame:
        """Combine subject features with demographic information"""
        try:
            # Merge subject features with demographic data
            merge_cols = ['subject_id', 'activity']
            available_merge_cols = [col for col in merge_cols if col in subject_features.columns and col in subjects_info.columns]
            
            if not available_merge_cols:
                logger.warning("No common columns for merging features and subject info")
                return subject_features
            
            demo_cols = ['gender', 'height', 'weight', 'age', 'bmi', 'gender_encoded', 'activity_encoded', 
                        'bp_sys_change', 'bp_dia_change', 'hr_1_change', 'hr_2_change', 'spo2_change']
            available_demo_cols = [col for col in demo_cols if col in subjects_info.columns]
            
            merge_subjects_info = subjects_info[available_merge_cols + available_demo_cols].drop_duplicates()
            
            integrated = subject_features.merge(
                merge_subjects_info,
                on=available_merge_cols,
                how='left'
            )
            
            # Create risk indicators for clot monitoring
            if 'bp_sys_change' in integrated.columns and 'bp_dia_change' in integrated.columns:
                integrated['bp_risk'] = (
                    (integrated['bp_sys_change'] > 20) | 
                    (integrated['bp_dia_change'] > 15)
                ).astype(int)
            
            if 'hr_1_change' in integrated.columns and 'hr_2_change' in integrated.columns:
                integrated['hr_variability_risk'] = (
                    (abs(integrated['hr_1_change']) > 30) | 
                    (abs(integrated['hr_2_change']) > 30)
                ).astype(int)
            
            if 'age' in integrated.columns:
                integrated['age_risk'] = (integrated['age'] > 50).astype(int)
                
            if 'bmi' in integrated.columns:
                integrated['bmi_risk'] = (integrated['bmi'] > 30).astype(int)
            
            # Composite risk score
            risk_cols = [col for col in ['bp_risk', 'hr_variability_risk', 'age_risk', 'bmi_risk'] 
                        if col in integrated.columns]
            
            if risk_cols:
                integrated['composite_risk_score'] = integrated[risk_cols].sum(axis=1)
            
            return integrated
            
        except Exception as e:
            logger.error(f"Error creating integrated features: {e}")
            return subject_features
    
    def _load_ppg_dataset(self) -> pd.DataFrame:
        """Load and preprocess the large PPG_Dataset.csv file"""
        try:
            ppg_file = self.csv_path / "PPG_Dataset.csv"
            
            if not ppg_file.exists():
                logger.warning(f"PPG_Dataset.csv not found at {ppg_file}")
                return pd.DataFrame()
            
            # Check file size
            file_size = ppg_file.stat().st_size / (1024 * 1024)  # Size in MB
            logger.info(f"Loading PPG_Dataset.csv ({file_size:.1f} MB)...")
            
            # Load dataset in chunks if it's very large (>500MB)
            if file_size > 500:
                logger.info("Large PPG dataset detected, using chunked loading...")
                chunks = []
                chunksize = 10000
                
                for i, chunk in enumerate(pd.read_csv(ppg_file, chunksize=chunksize)):
                    # Process each chunk
                    chunk_processed = self._preprocess_ppg_chunk(chunk)
                    chunks.append(chunk_processed)
                    
                    if i % 50 == 0:  # Log progress every 50 chunks
                        logger.info(f"Processed {i * chunksize:,} records...")
                        
                    # Optional: limit chunks for memory management
                    if len(chunks) > 200:  # Limit to ~2M records
                        logger.info("Reached chunk limit to prevent memory issues")
                        break
                
                ppg_data = pd.concat(chunks, ignore_index=True)
            else:
                # Load entire file at once for smaller files
                ppg_data = pd.read_csv(ppg_file)
                ppg_data = self._preprocess_ppg_chunk(ppg_data)
            
            logger.info(f"Successfully loaded PPG dataset: {len(ppg_data):,} records")
            return ppg_data
            
        except Exception as e:
            logger.error(f"Error loading PPG dataset: {e}")
            return pd.DataFrame()
    
    def _preprocess_ppg_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a chunk of PPG data"""
        try:
            # Clean column names
            chunk.columns = chunk.columns.str.strip()
            
            # Convert numeric columns
            numeric_cols = [col for col in chunk.columns if 'ppg' in col.lower() or 'pleth' in col.lower()]
            
            for col in numeric_cols:
                if col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Handle missing values in PPG signals
            chunk[numeric_cols] = chunk[numeric_cols].ffill().bfill().fillna(0)
            
            return chunk
            
        except Exception as e:
            logger.warning(f"Error preprocessing PPG chunk: {e}")
            return chunk
    
    def _extract_advanced_ppg_features(self, ppg_data: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced PPG features using ppg_analysis module"""
        try:
            from ppg_analysis import extract_ppg_features_from_dataset
            
            # Find PPG columns
            ppg_columns = [col for col in ppg_data.columns if 
                          'ppg' in col.lower() or 'pleth' in col.lower()]
            
            if not ppg_columns:
                logger.warning("No PPG columns found in dataset")
                return pd.DataFrame()
            
            logger.info(f"Extracting features from {len(ppg_columns)} PPG columns...")
            
            # Extract features using ppg_analysis module
            features = extract_ppg_features_from_dataset(
                data=ppg_data,
                ppg_columns=ppg_columns,
                window_size=2500  # 5 seconds at 500Hz
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting advanced PPG features: {e}")
            return pd.DataFrame()
    
    def save_processed_data(self, output_dir: str) -> None:
        """Save all processed data to files"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for key, data in self.integrated_data.items():
                if isinstance(data, pd.DataFrame) and not data.empty:
                    filepath = output_path / f"{key}.csv"
                    data.to_csv(filepath, index=False)
                    logger.info(f"Saved {key} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def generate_preprocessing_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of preprocessing results"""
        report = {
            'timestamp': pd.Timestamp.now(),
            'datasets_processed': list(self.integrated_data.keys()),
            'summary_statistics': {}
        }
        
        for key, data in self.integrated_data.items():
            if isinstance(data, pd.DataFrame):
                report['summary_statistics'][key] = {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'missing_values': int(data.isnull().sum().sum()),
                    'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024**2)
                }
        
        return report


def main():
    """Example usage of the preprocessing pipeline"""
    # Set up paths
    base_dir = Path(__file__).parent
    csv_path = base_dir / "csv"
    output_path = base_dir / "processed_data"
    
    if not csv_path.exists():
        logger.error(f"CSV directory not found: {csv_path}")
        return None
    
    # Initialize integrated preprocessor
    preprocessor = IntegratedPreprocessor(str(csv_path))
    
    # Run preprocessing
    results = preprocessor.run_complete_preprocessing(
        load_subjects=True,
        load_ppg_dataset=True, 
        max_subject_files=5  # Limit for demonstration
    )
    
    if not results:
        logger.error("No results from preprocessing")
        return None
    
    # Save results
    preprocessor.save_processed_data(str(output_path))
    
    # Generate report
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
    
    return results


if __name__ == "__main__":
    results = main()
