"""
PPG Signal Analysis and Feature Extraction - FIXED VERSION
==========================================

Simplified, working version for Photoplethysmography (PPG) signal processing
for clot monitoring and cardiovascular analysis.

Features:
- PPG signal quality assessment
- Heart rate variability analysis
- Blood flow pattern detection
- Anomaly detection for clot indicators

Author: AI Assistant
Project: AI Integration in Wearables for Clot Monitoring
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PPGSignalAnalyzer:
    """Simplified PPG signal analysis for clot monitoring"""
    
    def __init__(self, sampling_rate: float = 500.0):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def preprocess_ppg_signal(self, ppg_data: np.ndarray, 
                             lowcut: float = 0.5, 
                             highcut: float = 8.0) -> np.ndarray:
        """
        Preprocess PPG signal with bandpass filtering
        
        Args:
            ppg_data: Raw PPG signal as numpy array
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
        
        Returns:
            Filtered PPG signal as numpy array
        """
        try:
            # Ensure input is numpy array
            signal_array = np.asarray(ppg_data)
            
            # Design Butterworth bandpass filter
            low = lowcut / self.nyquist
            high = highcut / self.nyquist
            
            # Validate frequency ranges
            if low <= 0 or high >= 1 or low >= high:
                logger.warning("Invalid filter frequencies, returning original signal")
                return signal_array
            
            # Design Butterworth bandpass filter with robust error handling
            try:
                # Use SOS (Second Order Sections) format for better numerical stability
                if high > 0.9:  # Avoid Nyquist frequency issues
                    high = 0.9
                if low < 0.001:  # Avoid very low frequencies
                    low = 0.001
                
                sos = butter(4, [low, high], btype='band', output='sos')
                filtered_signal = signal.sosfilt(sos, signal_array)
                
                # Ensure we return numpy array
                return np.asarray(filtered_signal)
                
            except Exception as filter_error:
                logger.warning(f"Butterworth filter failed: {filter_error}, using median filter")
                filtered_signal = signal.medfilt(signal_array, kernel_size=5)
                return np.asarray(filtered_signal)
            
        except Exception as e:
            logger.warning(f"PPG preprocessing failed: {e}")
            return np.asarray(ppg_data)
    
    def assess_signal_quality(self, ppg_signal: np.ndarray, window_size: int = 1000) -> Dict[str, float]:
        """
        Assess PPG signal quality using multiple metrics
        
        Args:
            ppg_signal: PPG signal array
            window_size: Window size for analysis
        
        Returns:
            Dictionary containing quality metrics
        """
        try:
            signal_array = np.asarray(ppg_signal)
            quality_metrics = {}
            
            # Signal-to-noise ratio estimation
            signal_power = np.var(signal_array)
            noise_power = np.var(np.diff(signal_array))  # High-frequency noise approximation
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
            quality_metrics['snr_db'] = float(snr)
            
            # Perfusion index (PI) - measure of pulsatile strength
            ac_component = np.std(signal_array)
            dc_component = np.mean(signal_array)
            pi = (ac_component / dc_component) * 100 if dc_component > 0 else 0.0
            quality_metrics['perfusion_index'] = float(pi)
            
            # Correlation-based quality assessment
            if len(signal_array) > window_size * 2:
                windows = []
                for i in range(0, len(signal_array) - window_size, window_size):
                    windows.append(signal_array[i:i+window_size])
                
                if len(windows) >= 2:
                    correlations = []
                    for i in range(len(windows) - 1):
                        try:
                            corr, _ = pearsonr(windows[i], windows[i+1])
                            # Fix: Handle correlation result properly
                            if isinstance(corr, (int, float)) and not pd.isna(corr):
                                correlations.append(float(corr))
                        except:
                            continue
                    
                    quality_metrics['mean_correlation'] = float(np.mean(correlations)) if correlations else 0.0
                    quality_metrics['correlation_std'] = float(np.std(correlations)) if correlations else 0.0
                else:
                    quality_metrics['mean_correlation'] = 0.0
                    quality_metrics['correlation_std'] = 0.0
            else:
                quality_metrics['mean_correlation'] = 0.0
                quality_metrics['correlation_std'] = 0.0
            
            # Motion artifact detection (simplified)
            try:
                fft_signal = np.fft.fft(signal_array)
                high_freq_power = np.sum(np.abs(fft_signal)[int(len(fft_signal)*0.1):])
                total_power = np.sum(np.abs(fft_signal))
                quality_metrics['motion_artifact_ratio'] = float(high_freq_power / total_power if total_power > 0 else 0)
            except:
                quality_metrics['motion_artifact_ratio'] = 0.0
            
            # Overall quality score (0-100)
            quality_score = 0.0
            if snr > 10:
                quality_score += 25
            elif snr > 5:
                quality_score += 15
            
            if pi > 1:
                quality_score += 25
            elif pi > 0.5:
                quality_score += 15
            
            if quality_metrics['mean_correlation'] > 0.7:
                quality_score += 25
            elif quality_metrics['mean_correlation'] > 0.5:
                quality_score += 15
            
            if quality_metrics['motion_artifact_ratio'] < 0.3:
                quality_score += 25
            elif quality_metrics['motion_artifact_ratio'] < 0.5:
                quality_score += 15
            
            quality_metrics['overall_quality'] = float(quality_score)
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Signal quality assessment failed: {e}")
            return {
                'snr_db': 0.0,
                'perfusion_index': 0.0,
                'mean_correlation': 0.0,
                'correlation_std': 0.0,
                'motion_artifact_ratio': 1.0,
                'overall_quality': 0.0
            }
    
    def detect_peaks_and_valleys(self, ppg_signal: np.ndarray) -> Dict[str, Any]:
        """
        Detect peaks and valleys in PPG signal for pulse analysis
        
        Args:
            ppg_signal: Preprocessed PPG signal
        
        Returns:
            Dictionary with peak and valley information
        """
        try:
            signal_array = np.asarray(ppg_signal)
            
            # Find peaks (systolic points)
            min_distance = int(0.4 * self.fs)  # Minimum 0.4s between peaks
            height_threshold = np.percentile(signal_array, 60)
            prominence_threshold = np.std(signal_array) * 0.1
            
            peaks, peak_properties = find_peaks(
                signal_array, 
                height=height_threshold,
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            # Find valleys (diastolic points)
            valleys, valley_properties = find_peaks(
                -signal_array,
                height=-np.percentile(signal_array, 40),
                distance=min_distance,
                prominence=prominence_threshold
            )
            
            return {
                'peaks': peaks,
                'peak_heights': signal_array[peaks] if len(peaks) > 0 else np.array([]),
                'valleys': valleys,
                'valley_depths': signal_array[valleys] if len(valleys) > 0 else np.array([]),
                'num_peaks': len(peaks),
                'num_valleys': len(valleys)
            }
            
        except Exception as e:
            logger.error(f"Peak detection failed: {e}")
            return {
                'peaks': np.array([]),
                'peak_heights': np.array([]),
                'valleys': np.array([]),
                'valley_depths': np.array([]),
                'num_peaks': 0,
                'num_valleys': 0
            }
    
    def calculate_heart_rate_metrics(self, ppg_signal: np.ndarray) -> Dict[str, Any]:
        """
        Calculate heart rate and variability metrics
        
        Args:
            ppg_signal: PPG signal
        
        Returns:
            Dictionary with HR metrics
        """
        try:
            pulse_info = self.detect_peaks_and_valleys(ppg_signal)
            peaks = pulse_info['peaks']
            
            if len(peaks) < 2:
                return {'error': 'Insufficient peaks detected', 'num_peaks': len(peaks)}
            
            # Calculate RR intervals (peak-to-peak intervals)
            rr_intervals = np.diff(peaks) / self.fs  # Convert to seconds
            
            # Heart rate calculations
            hr_instantaneous = 60.0 / rr_intervals  # BPM for each interval
            hr_mean = float(np.mean(hr_instantaneous))
            hr_std = float(np.std(hr_instantaneous))
            
            # Basic HRV metrics
            rmssd = float(np.sqrt(np.mean(np.diff(rr_intervals) ** 2)))
            sdnn = float(np.std(rr_intervals))
            
            return {
                'mean_hr': hr_mean,
                'hr_std': hr_std,
                'min_hr': float(np.min(hr_instantaneous)),
                'max_hr': float(np.max(hr_instantaneous)),
                'num_peaks': len(peaks),
                'rmssd': rmssd,
                'sdnn': sdnn,
                'num_rr_intervals': len(rr_intervals)
            }
            
        except Exception as e:
            logger.error(f"Heart rate calculation failed: {e}")
            return {'error': str(e), 'num_peaks': 0}
    
    def detect_blood_flow_anomalies(self, ppg_signal: np.ndarray, 
                                  baseline_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Detect potential blood flow anomalies that could indicate clot formation
        
        Args:
            ppg_signal: PPG signal
            baseline_metrics: Optional baseline measurements for comparison
        
        Returns:
            Dictionary with anomaly indicators
        """
        try:
            # Current signal metrics
            current_metrics = self.assess_signal_quality(ppg_signal)
            hr_metrics = self.calculate_heart_rate_metrics(ppg_signal)
            
            anomaly_indicators = {
                'anomalies_detected': [],
                'risk_score': 0,
                'current_metrics': current_metrics
            }
            
            # Check for reduced perfusion (potential clot indicator)
            if current_metrics.get('perfusion_index', 0) < 0.5:
                anomaly_indicators['anomalies_detected'].append('low_perfusion')
                anomaly_indicators['risk_score'] += 2
            
            # Check for irregular heart rate patterns
            if 'hr_std' in hr_metrics and hr_metrics['hr_std'] > 15:
                anomaly_indicators['anomalies_detected'].append('high_hr_variability')
                anomaly_indicators['risk_score'] += 1
            
            # Check signal quality degradation
            if current_metrics.get('overall_quality', 0) < 50:
                anomaly_indicators['anomalies_detected'].append('poor_signal_quality')
                anomaly_indicators['risk_score'] += 1
            
            # Compare with baseline if provided
            if baseline_metrics:
                baseline_pi = baseline_metrics.get('perfusion_index', 0)
                current_pi = current_metrics.get('perfusion_index', 0)
                
                # Significant decrease in perfusion index
                if baseline_pi > 0 and current_pi > 0 and (current_pi / baseline_pi) < 0.7:
                    anomaly_indicators['anomalies_detected'].append('perfusion_decrease')
                    anomaly_indicators['risk_score'] += 3
                
                # Check for correlation decrease
                baseline_corr = baseline_metrics.get('mean_correlation', 0)
                current_corr = current_metrics.get('mean_correlation', 0)
                
                if baseline_corr > 0.5 and current_corr < 0.3:
                    anomaly_indicators['anomalies_detected'].append('signal_correlation_drop')
                    anomaly_indicators['risk_score'] += 2
            
            # Overall risk assessment
            risk_score = anomaly_indicators['risk_score']
            if risk_score >= 5:
                anomaly_indicators['risk_level'] = 'HIGH'
            elif risk_score >= 3:
                anomaly_indicators['risk_level'] = 'MEDIUM'
            elif risk_score >= 1:
                anomaly_indicators['risk_level'] = 'LOW'
            else:
                anomaly_indicators['risk_level'] = 'NORMAL'
            
            return anomaly_indicators
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {
                'anomalies_detected': ['analysis_error'],
                'risk_score': 0,
                'risk_level': 'UNKNOWN',
                'error': str(e)
            }
    
    def analyze_ppg_window(self, ppg_data: np.ndarray, 
                          ecg_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of a PPG signal window
        
        Args:
            ppg_data: PPG signal window
            ecg_data: Optional ECG signal window for advanced analysis
        
        Returns:
            Complete analysis results
        """
        try:
            # Preprocess signal
            preprocessed_ppg = self.preprocess_ppg_signal(ppg_data)
            
            # Comprehensive analysis
            results = {
                'signal_length': len(ppg_data),
                'sampling_rate': self.fs,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Signal quality assessment
            results['quality_metrics'] = self.assess_signal_quality(preprocessed_ppg)
            
            # Heart rate analysis
            results['hr_metrics'] = self.calculate_heart_rate_metrics(preprocessed_ppg)
            
            # Pulse characteristics
            results['pulse_info'] = self.detect_peaks_and_valleys(preprocessed_ppg)
            
            # Anomaly detection
            results['anomaly_analysis'] = self.detect_blood_flow_anomalies(preprocessed_ppg)
            
            return results
            
        except Exception as e:
            logger.error(f"PPG window analysis failed: {e}")
            return {
                'error': str(e),
                'signal_length': len(ppg_data) if ppg_data is not None else 0,
                'sampling_rate': self.fs
            }


def extract_ppg_features_from_dataset(data: pd.DataFrame, 
                                    ppg_columns: List[str],
                                    ecg_columns: Optional[List[str]] = None,
                                    window_size: int = 2500) -> pd.DataFrame:
    """
    Extract comprehensive PPG features from dataset
    
    Args:
        data: DataFrame with PPG (and optionally ECG) data
        ppg_columns: List of PPG column names
        ecg_columns: Optional list of ECG column names
        window_size: Window size for analysis
    
    Returns:
        DataFrame with extracted features
    """
    try:
        analyzer = PPGSignalAnalyzer()
        features_list = []
        
        logger.info(f"Processing {len(ppg_columns)} PPG columns with window size {window_size}")
        
        for i in range(0, len(data) - window_size, window_size):
            window_data = data.iloc[i:i+window_size]
            
            for ppg_col in ppg_columns:
                if ppg_col not in window_data.columns:
                    logger.warning(f"PPG column {ppg_col} not found in data")
                    continue
                    
                try:
                    ppg_signal = np.asarray(window_data[ppg_col].values)
                    
                    # Skip if signal is too short or contains only NaN
                    if len(ppg_signal) < 100 or np.all(np.isnan(ppg_signal)):
                        continue
                    
                    # Replace NaN with interpolated values
                    if np.any(np.isnan(ppg_signal)):
                        valid_indices = ~np.isnan(ppg_signal)
                        if np.sum(valid_indices) > 10:  # Need at least 10 valid points
                            ppg_signal = np.interp(
                                np.arange(len(ppg_signal)),
                                np.arange(len(ppg_signal))[valid_indices],
                                ppg_signal[valid_indices]
                            )
                        else:
                            continue
                    
                    # Analyze window
                    analysis = analyzer.analyze_ppg_window(ppg_signal)
                    
                    if 'error' in analysis:
                        logger.warning(f"Analysis failed for window {i//window_size}, column {ppg_col}: {analysis['error']}")
                        continue
                    
                    # Create feature dictionary
                    feature_dict = {
                        'window_id': i // window_size,
                        'ppg_channel': ppg_col,
                        'signal_length': analysis['signal_length']
                    }
                    
                    # Add quality metrics
                    for key, value in analysis['quality_metrics'].items():
                        feature_dict[f'quality_{key}'] = value
                    
                    # Add HR metrics
                    if 'error' not in analysis['hr_metrics']:
                        for key, value in analysis['hr_metrics'].items():
                            if key not in ['rr_intervals']:
                                feature_dict[f'hr_{key}'] = value
                    
                    # Add pulse info
                    pulse_info = analysis['pulse_info']
                    feature_dict['pulse_num_peaks'] = pulse_info['num_peaks']
                    feature_dict['pulse_num_valleys'] = pulse_info['num_valleys']
                    
                    # Add anomaly metrics
                    anomaly = analysis['anomaly_analysis']
                    feature_dict['anomaly_risk_score'] = anomaly['risk_score']
                    feature_dict['anomaly_risk_level'] = anomaly['risk_level']
                    feature_dict['num_anomalies'] = len(anomaly['anomalies_detected'])
                    
                    features_list.append(feature_dict)
                    
                except Exception as e:
                    logger.warning(f"Error processing window {i//window_size}, column {ppg_col}: {e}")
                    continue
        
        if not features_list:
            logger.warning("No features extracted from dataset")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(features_list)
        logger.info(f"Successfully extracted {len(result_df)} feature records")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    print("PPG Signal Analyzer - Fixed Version")
    print("This module provides PPG analysis capabilities including:")
    print("- Signal quality assessment")
    print("- Heart rate variability analysis")
    print("- Blood flow anomaly detection")
    print("- Clot risk indicators")
    
    # Simple test
    try:
        analyzer = PPGSignalAnalyzer()
        test_signal = np.sin(2 * np.pi * 1.2 * np.linspace(0, 10, 5000)) + 0.1 * np.random.randn(5000)
        
        analysis = analyzer.analyze_ppg_window(test_signal)
        print("\nTest analysis completed successfully:")
        print(f"- Quality score: {analysis['quality_metrics']['overall_quality']:.1f}")
        print(f"- Number of peaks: {analysis['pulse_info']['num_peaks']}")
        print(f"- Risk level: {analysis['anomaly_analysis']['risk_level']}")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
