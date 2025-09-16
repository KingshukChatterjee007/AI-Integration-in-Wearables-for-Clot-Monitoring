"""
RRest-Syn Dataset Preprocessing
==============================

Specialized preprocessing for the RRest-Syn synthetic respiratory and cardiac signals dataset.
This dataset contains simulated PPG (Pleth) and ECG (II) signals with various respiratory 
modulation patterns for clot monitoring and cardiovascular analysis.

Features:
- Respiratory signal analysis
- PPG signal quality assessment  
- Heart rate variability analysis
- Respiratory pattern extraction
- Feature engineering for ML models

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
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.stats import zscore
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RRestSynAnalyzer:
    """Analyzer for RRest-Syn synthetic respiratory and cardiac signals"""
    
    def __init__(self, sampling_rate: float = 500.0):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def preprocess_signal(self, signal_data: np.ndarray, 
                         signal_type: str = 'ppg',
                         lowcut: float = 0.5, 
                         highcut: float = 40.0) -> np.ndarray:
        """
        Preprocess respiratory/cardiac signal with appropriate filtering
        
        Args:
            signal_data: Raw signal as numpy array
            signal_type: 'ppg' for pleth signals, 'ecg' for ECG signals
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
        
        Returns:
            Filtered signal as numpy array
        """
        try:
            signal_array = np.asarray(signal_data)
            
            # Adjust filter parameters based on signal type
            if signal_type.lower() == 'ecg':
                lowcut = 0.5
                highcut = 50.0  # ECG typically needs higher frequency content
            elif signal_type.lower() == 'ppg' or signal_type.lower() == 'pleth':
                lowcut = 0.5
                highcut = 20.0  # PPG signals typically lower frequency
            
            # Design Butterworth bandpass filter
            low = lowcut / self.nyquist
            high = highcut / self.nyquist
            
            # Validate frequency ranges
            if low <= 0 or high >= 1 or low >= high:
                logger.warning("Invalid filter frequencies, returning original signal")
                return signal_array
            
            # Apply robust filtering
            try:
                if high > 0.95:
                    high = 0.95
                if low < 0.001:
                    low = 0.001
                
                sos = butter(4, [low, high], btype='band', output='sos')
                filtered_signal = signal.sosfilt(sos, signal_array)
                
                return np.asarray(filtered_signal)
                
            except Exception as filter_error:
                logger.warning(f"Butterworth filter failed: {filter_error}, using median filter")
                filtered_signal = signal.medfilt(signal_array, kernel_size=5)
                return np.asarray(filtered_signal)
            
        except Exception as e:
            logger.warning(f"Signal preprocessing failed: {e}")
            return np.asarray(signal_data)
    
    def extract_respiratory_features(self, signal_data: np.ndarray, 
                                   expected_rr: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract respiratory features from the signal
        
        Args:
            signal_data: Preprocessed signal
            expected_rr: Expected respiratory rate (breaths/min) from metadata
        
        Returns:
            Dictionary with respiratory features
        """
        try:
            features = {}
            signal_array = np.asarray(signal_data)
            
            # Respiratory rate estimation using spectral analysis
            # Focus on typical respiratory frequency range (0.1-1.5 Hz = 6-90 breaths/min)
            fft_signal = np.fft.fft(signal_array)
            freqs = np.fft.fftfreq(len(signal_array), 1/self.fs)
            
            # Focus on respiratory frequency range
            resp_freq_mask = (freqs >= 0.1) & (freqs <= 1.5)
            resp_power_spectrum = np.abs(fft_signal[resp_freq_mask])
            resp_freqs = freqs[resp_freq_mask]
            
            if len(resp_power_spectrum) > 0:
                # Find dominant respiratory frequency
                dominant_freq_idx = np.argmax(resp_power_spectrum)
                estimated_rr_hz = resp_freqs[dominant_freq_idx]
                estimated_rr_bpm = estimated_rr_hz * 60
                
                features['estimated_respiratory_rate_bpm'] = float(estimated_rr_bpm)
                features['respiratory_power'] = float(np.max(resp_power_spectrum))
                features['respiratory_frequency_hz'] = float(estimated_rr_hz)
            else:
                features['estimated_respiratory_rate_bpm'] = 0.0
                features['respiratory_power'] = 0.0
                features['respiratory_frequency_hz'] = 0.0
            
            # Respiratory variability measures
            # Use envelope analysis for respiratory pattern
            try:
                # Extract respiratory envelope using Hilbert transform
                analytic_signal = signal.hilbert(signal_array)
                # Convert to numpy array if it's not already one
                analytic_signal = np.asarray(analytic_signal)
                amplitude_envelope = np.abs(analytic_signal)
                
                # Respiratory modulation depth
                features['respiratory_modulation_depth'] = float(np.std(amplitude_envelope))
                features['respiratory_envelope_mean'] = float(np.mean(amplitude_envelope))
                features['respiratory_envelope_std'] = float(np.std(amplitude_envelope))
                
                # Respiratory pattern regularity (using autocorrelation)
                autocorr = np.correlate(amplitude_envelope, amplitude_envelope, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                
                # Find peaks in autocorrelation to assess periodicity
                if len(autocorr) > 100:
                    peaks, _ = find_peaks(autocorr[50:], height=np.max(autocorr)*0.3)
                    if len(peaks) > 0:
                        # Respiratory pattern regularity score
                        features['respiratory_regularity'] = float(np.max(autocorr[50:][peaks]) / np.max(autocorr))
                    else:
                        features['respiratory_regularity'] = 0.0
                else:
                    features['respiratory_regularity'] = 0.0
                    
            except Exception as e:
                logger.debug(f"Respiratory envelope analysis failed: {e}")
                features['respiratory_modulation_depth'] = 0.0
                features['respiratory_envelope_mean'] = 0.0
                features['respiratory_envelope_std'] = 0.0
                features['respiratory_regularity'] = 0.0
            
            # Compare with expected respiratory rate if provided
            if expected_rr is not None:
                features['expected_respiratory_rate_bpm'] = float(expected_rr)
                estimated_rr = features.get('estimated_respiratory_rate_bpm', 0.0)
                if estimated_rr > 0:
                    features['respiratory_rate_accuracy'] = float(1.0 - abs(estimated_rr - expected_rr) / expected_rr)
                else:
                    features['respiratory_rate_accuracy'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Respiratory feature extraction failed: {e}")
            return {
                'estimated_respiratory_rate_bpm': 0.0,
                'respiratory_power': 0.0,
                'respiratory_frequency_hz': 0.0,
                'respiratory_modulation_depth': 0.0,
                'respiratory_envelope_mean': 0.0,
                'respiratory_envelope_std': 0.0,
                'respiratory_regularity': 0.0
            }
    
    def extract_cardiac_features(self, signal_data: np.ndarray,
                               expected_hr: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract cardiac features from PPG or ECG signal
        
        Args:
            signal_data: Preprocessed signal
            expected_hr: Expected heart rate (beats/min) from metadata
        
        Returns:
            Dictionary with cardiac features
        """
        try:
            features = {}
            signal_array = np.asarray(signal_data)
            
            # Heart rate estimation using peak detection
            # Find peaks (R-waves for ECG, systolic peaks for PPG)
            min_distance = int(0.4 * self.fs)  # Minimum 0.4s between beats (150 BPM max)
            height_threshold = np.percentile(signal_array, 70)
            prominence_threshold = np.std(signal_array) * 0.2
            
            peaks, peak_properties = find_peaks(
                signal_array,
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            if len(peaks) >= 2:
                # Calculate RR intervals
                rr_intervals = np.diff(peaks) / self.fs  # Convert to seconds
                hr_instantaneous = 60.0 / rr_intervals  # BPM for each interval
                
                features['mean_heart_rate_bpm'] = float(np.mean(hr_instantaneous))
                features['heart_rate_std'] = float(np.std(hr_instantaneous))
                features['min_heart_rate'] = float(np.min(hr_instantaneous))
                features['max_heart_rate'] = float(np.max(hr_instantaneous))
                features['num_beats'] = len(peaks)
                
                # HRV metrics
                features['rmssd'] = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))  # RMSSD
                features['sdnn'] = float(np.std(rr_intervals))  # SDNN
                features['pnn50'] = float(np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals) * 100)
                
                # Heart rate variability in frequency domain
                if len(rr_intervals) > 10:
                    # Interpolate RR intervals for frequency analysis
                    time_rr = np.cumsum(rr_intervals)
                    time_interp = np.linspace(0, time_rr[-1], int(time_rr[-1] * 4))  # 4 Hz sampling
                    rr_interp = np.interp(time_interp, time_rr, rr_intervals)
                    
                    # FFT for HRV frequency analysis
                    fft_rr = np.fft.fft(rr_interp - np.mean(rr_interp))
                    freqs_rr = np.fft.fftfreq(len(rr_interp), 1/4.0)
                    power_rr = np.abs(fft_rr) ** 2
                    
                    # VLF, LF, HF power bands
                    vlf_mask = (freqs_rr >= 0.003) & (freqs_rr < 0.04)
                    lf_mask = (freqs_rr >= 0.04) & (freqs_rr < 0.15)
                    hf_mask = (freqs_rr >= 0.15) & (freqs_rr < 0.4)
                    
                    features['vlf_power'] = float(np.sum(power_rr[vlf_mask]))
                    features['lf_power'] = float(np.sum(power_rr[lf_mask]))
                    features['hf_power'] = float(np.sum(power_rr[hf_mask]))
                    
                    total_power = features['vlf_power'] + features['lf_power'] + features['hf_power']
                    if total_power > 0:
                        features['lf_hf_ratio'] = float(features['lf_power'] / features['hf_power']) if features['hf_power'] > 0 else 0.0
                    else:
                        features['lf_hf_ratio'] = 0.0
                else:
                    features['vlf_power'] = 0.0
                    features['lf_power'] = 0.0
                    features['hf_power'] = 0.0
                    features['lf_hf_ratio'] = 0.0
                
            else:
                # Insufficient peaks detected
                features.update({
                    'mean_heart_rate_bpm': 0.0,
                    'heart_rate_std': 0.0,
                    'min_heart_rate': 0.0,
                    'max_heart_rate': 0.0,
                    'num_beats': len(peaks),
                    'rmssd': 0.0,
                    'sdnn': 0.0,
                    'pnn50': 0.0,
                    'vlf_power': 0.0,
                    'lf_power': 0.0,
                    'hf_power': 0.0,
                    'lf_hf_ratio': 0.0
                })
            
            # Compare with expected heart rate if provided
            if expected_hr is not None:
                features['expected_heart_rate_bpm'] = float(expected_hr)
                estimated_hr = features.get('mean_heart_rate_bpm', 0.0)
                if estimated_hr > 0:
                    features['heart_rate_accuracy'] = float(1.0 - abs(estimated_hr - expected_hr) / expected_hr)
                else:
                    features['heart_rate_accuracy'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Cardiac feature extraction failed: {e}")
            return {
                'mean_heart_rate_bpm': 0.0,
                'heart_rate_std': 0.0,
                'min_heart_rate': 0.0,
                'max_heart_rate': 0.0,
                'num_beats': 0,
                'rmssd': 0.0,
                'sdnn': 0.0,
                'pnn50': 0.0,
                'vlf_power': 0.0,
                'lf_power': 0.0,
                'hf_power': 0.0,
                'lf_hf_ratio': 0.0
            }
    
    def extract_signal_quality_features(self, signal_data: np.ndarray) -> Dict[str, Any]:
        """Extract signal quality and statistical features"""
        try:
            signal_array = np.asarray(signal_data)
            features = {}
            
            # Basic statistical features
            features['signal_mean'] = float(np.mean(signal_array))
            features['signal_std'] = float(np.std(signal_array))
            features['signal_min'] = float(np.min(signal_array))
            features['signal_max'] = float(np.max(signal_array))
            features['signal_range'] = features['signal_max'] - features['signal_min']
            
            # Use absolute values of the signal for statistical calculations
            signal_abs = np.abs(signal_array)  # Get magnitude of potentially complex values
            
            # Calculate skewness and kurtosis using scipy.stats for better reliability
            try:
                from scipy.stats import skew, kurtosis
                features['signal_skewness'] = float(skew(signal_abs))
            except (ValueError, TypeError, ImportError):
                features['signal_skewness'] = 0.0
            
            try:
                from scipy.stats import skew, kurtosis
                features['signal_kurtosis'] = float(kurtosis(signal_abs))
            except (ValueError, TypeError, ImportError):
                features['signal_kurtosis'] = 0.0
            
            # Signal quality metrics
            # SNR estimation
            signal_power = np.var(signal_array)
            noise_power = np.var(np.diff(signal_array))
            features['snr_db'] = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else 20.0
            
            # Signal stationarity (using variance in sliding windows)
            window_size = len(signal_array) // 10
            if window_size > 100:
                window_vars = []
                for i in range(0, len(signal_array) - window_size, window_size):
                    window_vars.append(np.var(signal_array[i:i+window_size]))
                features['signal_stationarity'] = float(1.0 - np.std(window_vars) / np.mean(window_vars)) if np.mean(window_vars) > 0 else 0.0
            else:
                features['signal_stationarity'] = 1.0
            
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.signbit(signal_array - np.mean(signal_array))))[0]
            features['zero_crossing_rate'] = float(len(zero_crossings) / len(signal_array))
            
            return features
            
        except Exception as e:
            logger.error(f"Signal quality feature extraction failed: {e}")
            return {
                'signal_mean': 0.0,
                'signal_std': 0.0,
                'signal_min': 0.0,
                'signal_max': 0.0,
                'signal_range': 0.0,
                'signal_skewness': 0.0,
                'signal_kurtosis': 0.0,
                'snr_db': 0.0,
                'signal_stationarity': 0.0,
                'zero_crossing_rate': 0.0
            }
    
    def validate_signal(self, signal_data, name="signal"):
        """Helper function to validate signal data"""
        signal_array = np.asarray(signal_data)
        logger.debug(f"{name} shape: {signal_array.shape}, dtype: {signal_array.dtype}")
        logger.debug(f"{name} contains complex: {np.iscomplexobj(signal_array)}")
        logger.debug(f"{name} range: {np.min(np.abs(signal_array))} to {np.max(np.abs(signal_array))}")
        return signal_array


class RRestSynPreprocessor:
    """Main preprocessor for RRest-Syn dataset"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.analyzer = RRestSynAnalyzer()
        self.processed_data = {}
        
    def parse_metadata_file(self, fix_file_path: Path) -> Dict[str, Any]:
        """Parse the _fix.txt metadata file to extract signal parameters"""
        try:
            metadata = {}
            
            with open(fix_file_path, 'r') as f:
                content = f.read()
            
            # Extract information using regex patterns
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if 'Signals:' in line:
                    # Extract signal types
                    signals_part = line.split('Signals:')[1].strip()
                    metadata['signal_description'] = signals_part
                    
                elif 'Sampling frequency:' in line:
                    # Extract sampling frequency
                    freq_match = re.search(r'(\d+)\s*Hz', line)
                    if freq_match:
                        metadata['sampling_frequency'] = int(freq_match.group(1))
                        
                elif 'Respiratory modulation type:' in line:
                    # Extract respiratory modulation type
                    mod_type = line.split('type:')[1].strip()
                    metadata['respiratory_modulation_type'] = mod_type
                    
                elif 'Simulated respiratory rate:' in line:
                    # Extract respiratory rate
                    rr_match = re.search(r'(\d+\.?\d*)\s*breaths/min', line)
                    if rr_match:
                        metadata['simulated_respiratory_rate_bpm'] = float(rr_match.group(1))
                        
                elif 'Simulated heart rate:' in line:
                    # Extract heart rate
                    hr_match = re.search(r'(\d+\.?\d*)\s*beats/min', line)
                    if hr_match:
                        metadata['simulated_heart_rate_bpm'] = float(hr_match.group(1))
            
            # Extract subject/file ID from filename
            file_stem = fix_file_path.stem.replace('_fix', '')
            metadata['file_id'] = file_stem
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to parse metadata from {fix_file_path}: {e}")
            return {'file_id': fix_file_path.stem.replace('_fix', '')}
    
    def process_single_file(self, data_file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single RRest-Syn data file"""
        try:
            # Load signal data
            data = pd.read_csv(data_file_path, header=None, names=['time', 'signal_value'])
            
            if len(data) < 1000:  # Skip files that are too short
                logger.warning(f"File {data_file_path.name} too short: {len(data)} samples")
                return None
            
            # Load corresponding metadata
            fix_file_path = data_file_path.parent / f"{data_file_path.stem.replace('_data', '_fix')}.txt"
            metadata = self.parse_metadata_file(fix_file_path) if fix_file_path.exists() else {}
            
            # Extract signal array
            signal_values = data['signal_value'].values
            time_values = data['time'].values
            
            # Preprocess signal (assume PPG-like characteristics for now)
            signal_type = 'ppg'  # Most rrest-syn files are pleth (PPG) signals
            # Explicitly convert to numpy array before passing to the function
            signal_values_array = np.asarray(signal_values)
            preprocessed_signal = self.analyzer.preprocess_signal(signal_values_array, signal_type)
            
            # Extract comprehensive features
            features = {
                'file_id': metadata.get('file_id', data_file_path.stem),
                'file_path': str(data_file_path),
                'signal_length': len(signal_values),
                'duration_seconds': float(time_values[-1] - time_values[0]) if len(time_values) > 1 else 0.0,
                'sampling_rate': metadata.get('sampling_frequency', 500)
            }
            
            # Add metadata features
            features.update(metadata)
            
            # Extract respiratory features
            expected_rr = metadata.get('simulated_respiratory_rate_bpm')
            respiratory_features = self.analyzer.extract_respiratory_features(preprocessed_signal, expected_rr)
            features.update({f'resp_{k}': v for k, v in respiratory_features.items()})
            
            # Extract cardiac features
            expected_hr = metadata.get('simulated_heart_rate_bpm')
            cardiac_features = self.analyzer.extract_cardiac_features(preprocessed_signal, expected_hr)
            features.update({f'cardiac_{k}': v for k, v in cardiac_features.items()})
            
            # Extract signal quality features
            quality_features = self.analyzer.extract_signal_quality_features(preprocessed_signal)
            features.update({f'quality_{k}': v for k, v in quality_features.items()})
            
            # Add processing timestamp
            features['processed_timestamp'] = pd.Timestamp.now().isoformat()
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to process {data_file_path}: {e}")
            return None
    
    def process_all_files(self, max_files: Optional[int] = None, 
                         use_parallel: bool = True, 
                         n_workers: int = 4) -> pd.DataFrame:
        """Process all RRest-Syn files"""
        try:
            # Find all data files
            data_files = list(self.data_path.glob('*_data.csv'))
            
            if max_files:
                data_files = data_files[:max_files]
            
            logger.info(f"Processing {len(data_files)} RRest-Syn files...")
            
            all_features = []
            
            if use_parallel and len(data_files) > 10:
                # Process files in parallel
                logger.info(f"Using parallel processing with {n_workers} workers")
                
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all jobs
                    futures = {executor.submit(self.process_single_file, file_path): file_path 
                              for file_path in data_files}
                    
                    # Collect results
                    for i, future in enumerate(as_completed(futures)):
                        try:
                            result = future.result(timeout=60)  # 60 second timeout per file
                            if result is not None:
                                all_features.append(result)
                            
                            # Progress logging
                            if (i + 1) % 20 == 0:
                                logger.info(f"Processed {i + 1}/{len(data_files)} files...")
                                
                        except Exception as e:
                            file_path = futures[future]
                            logger.error(f"Error processing {file_path.name}: {e}")
                            continue
            else:
                # Process files sequentially
                for i, file_path in enumerate(data_files):
                    try:
                        result = self.process_single_file(file_path)
                        if result is not None:
                            all_features.append(result)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"Processed {i + 1}/{len(data_files)} files...")
                            
                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")
                        continue
            
            if not all_features:
                logger.error("No features extracted from any files")
                return pd.DataFrame()
            
            # Create DataFrame
            features_df = pd.DataFrame(all_features)
            
            # Clean up any infinite or invalid values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(0.0)
            
            logger.info(f"Successfully processed {len(features_df)} files with {len(features_df.columns)} features each")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to process RRest-Syn files: {e}")
            return pd.DataFrame()
    
    def save_processed_data(self, features_df: pd.DataFrame, output_path: str) -> None:
        """Save processed features to CSV"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove file_path and processed_timestamp columns before saving
            if 'file_path' in features_df.columns:
                features_df = features_df.drop(columns=['file_path'])
                
            if 'processed_timestamp' in features_df.columns:
                features_df = features_df.drop(columns=['processed_timestamp'])
            
            features_df.to_csv(output_file, index=False)
            logger.info(f"Saved RRest-Syn processed data to {output_file}")
            logger.info(f"Shape: {features_df.shape}")
            logger.info(f"Columns removed: file_path, processed_timestamp")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
    
    def generate_summary_report(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary report of processing results"""
        try:
            report = {
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'total_files_processed': len(features_df),
                'total_features_extracted': len(features_df.columns),
                'data_quality_metrics': {}
            }
            
            # Basic statistics
            if not features_df.empty:
                numeric_columns = features_df.select_dtypes(include=[np.number]).columns
                
                report['data_quality_metrics'] = {
                    'mean_signal_length': float(features_df['signal_length'].mean()) if 'signal_length' in features_df.columns else 0,
                    'mean_duration_seconds': float(features_df['duration_seconds'].mean()) if 'duration_seconds' in features_df.columns else 0,
                    'mean_snr_db': float(features_df['quality_snr_db'].mean()) if 'quality_snr_db' in features_df.columns else 0,
                    'respiratory_rate_range': {
                        'min': float(features_df['resp_estimated_respiratory_rate_bpm'].min()) if 'resp_estimated_respiratory_rate_bpm' in features_df.columns else 0,
                        'max': float(features_df['resp_estimated_respiratory_rate_bpm'].max()) if 'resp_estimated_respiratory_rate_bpm' in features_df.columns else 0
                    },
                    'heart_rate_range': {
                        'min': float(features_df['cardiac_mean_heart_rate_bpm'].min()) if 'cardiac_mean_heart_rate_bpm' in features_df.columns else 0,
                        'max': float(features_df['cardiac_mean_heart_rate_bpm'].max()) if 'cardiac_mean_heart_rate_bpm' in features_df.columns else 0
                    }
                }
                
                # Respiratory modulation types
                if 'respiratory_modulation_type' in features_df.columns:
                    report['respiratory_modulation_types'] = features_df['respiratory_modulation_type'].value_counts().to_dict()
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return {'error': str(e)}


def main():
    """Main function to run RRest-Syn preprocessing"""
    try:
        # Set up paths
        base_dir = Path(__file__).parent
        rrest_syn_path = base_dir.parent / "rrest-syn_csv"
        output_path = base_dir.parent / "processed_data" / "rrest_syn_features.csv"
        
        if not rrest_syn_path.exists():
            logger.error(f"RRest-Syn directory not found: {rrest_syn_path}")
            return None
        
        # Initialize preprocessor
        preprocessor = RRestSynPreprocessor(str(rrest_syn_path))
        
        # Process all files (or limit for testing)
        logger.info("Starting RRest-Syn dataset preprocessing...")
        features_df = preprocessor.process_all_files(
            max_files=None,  # Process all files
            use_parallel=True,  # Use parallel processing
            n_workers=4  # Number of parallel workers
        )
        
        if features_df.empty:
            logger.error("No data was processed successfully")
            return None
        
        # Save processed data
        preprocessor.save_processed_data(features_df, str(output_path))
        
        # Generate and display summary report
        report = preprocessor.generate_summary_report(features_df)
        
        print("\n" + "="*70)
        print("RREST-SYN PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Files processed: {report['total_files_processed']}")
        print(f"Features extracted: {report['total_features_extracted']}")
        print(f"Mean signal length: {report['data_quality_metrics'].get('mean_signal_length', 0):.0f} samples")
        print(f"Mean duration: {report['data_quality_metrics'].get('mean_duration_seconds', 0):.1f} seconds")
        print(f"Mean SNR: {report['data_quality_metrics'].get('mean_snr_db', 0):.1f} dB")
        
        if 'respiratory_modulation_types' in report:
            print(f"Respiratory modulation types: {report['respiratory_modulation_types']}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Main preprocessing failed: {e}")
        return None


if __name__ == "__main__":
    results = main()