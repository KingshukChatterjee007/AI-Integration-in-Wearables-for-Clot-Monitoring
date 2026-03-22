"""
Wearable Exam Stress Dataset Preprocessor (Empatica E4 format)
==============================================================
Extracts features from ACC, BVP, EDA, HR, IBI, and TEMP files.
Aligns these multi-rate signals into coherent 60-second windows.

Author: AI Assistant
Project: AI Integration in Wearables for Clot Monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def process_empatica_file(filepath):
    """
    Reads an Empatica E4 CSV format.
    Row 0: Start Unix Timestamp
    Row 1: Sampling Frequency (Hz)
    Row 2+: Signal Values
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        if len(lines) < 3:
            return None, None, None
            
        start_ts = float(lines[0].strip().split(',')[0])
        freq = float(lines[1].strip().split(',')[0])
        
        # Read the rest as a numpy array
        # ACC has 3 columns (x, y, z), others have 1
        data = np.loadtxt(lines[2:], delimiter=',')
        
        # Generate timestamps for each data point
        num_samples = data.shape[0]
        timestamps = start_ts + np.arange(num_samples) / freq
        
        return timestamps, data, freq
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None, None, None

def extract_window_features(signal_data, prefix, freq):
    """Extract statistical and clinical features from a signal window."""
    if len(signal_data) == 0:
        return {}
        
    features = {
        f'{prefix}_mean': np.mean(signal_data),
        f'{prefix}_std': np.std(signal_data),
        f'{prefix}_min': np.min(signal_data),
        f'{prefix}_max': np.max(signal_data),
        f'{prefix}_range': np.max(signal_data) - np.min(signal_data)
    }
    
    # Advanced stats if enough data
    if len(signal_data) > 3:
        features[f'{prefix}_skew'] = skew(signal_data)
        features[f'{prefix}_kurt'] = kurtosis(signal_data)
        
    # Specific EDA features (Galvanic Skin Response)
    if prefix == 'eda':
        # Simple peak detection (skin conductance responses)
        # A simple derivative-based approach for demonstration
        diff_eda = np.diff(signal_data)
        num_peaks = np.sum(diff_eda > 0.01) # arbitrary threshold for steep rise
        features['eda_peaks_count'] = num_peaks
        
    return features

def extract_acc_features(acc_data, freq):
    """Specific processing for 3-axis Accelerometer data."""
    if len(acc_data) == 0 or len(acc_data.shape) != 2 or acc_data.shape[1] != 3:
        return {}
        
    # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
    # Note: Empatica ACC is measured in units of 1/64g. 
    # For relative magnitude features, the raw unit is fine as long as scaled later.
    x, y, z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    
    return {
        'acc_mag_mean': np.mean(magnitude),
        'acc_mag_std': np.std(magnitude),
        'acc_x_var': np.var(x),
        'acc_y_var': np.var(y),
        'acc_z_var': np.var(z)
    }

def process_session(session_path, subject_id, exam_type, window_size_sec=60):
    """
    Process all sensor files in a session directory (e.g., S1/Final)
    and align them into fixed-size time windows.
    """
    logger.info(f"Processing {subject_id} - {exam_type}")
    
    files = {
        'bvp': 'BVP.csv',
        'eda': 'EDA.csv',
        'hr': 'HR.csv',
        'temp': 'TEMP.csv',
        'acc': 'ACC.csv'
    }
    
    signals = {}
    start_times = {}
    end_times = {}
    
    # 1. Load all available signals
    for key, filename in files.items():
        filepath = session_path / filename
        if not filepath.exists():
            continue
            
        ts, data, freq = process_empatica_file(filepath)
        if ts is not None and len(ts) > 0:
            signals[key] = {'ts': ts, 'data': data, 'freq': freq}
            start_times[key] = ts[0]
            end_times[key] = ts[-1]
            
    if not signals:
        return pd.DataFrame()
        
    # 2. Find common time overlap
    global_start = max(start_times.values())
    global_end = min(end_times.values())
    
    if global_start >= global_end:
        return pd.DataFrame()
        
    # 3. Create synchronized windows
    num_windows = int((global_end - global_start) // window_size_sec)
    windows_data = []
    
    for i in range(num_windows):
        win_start = global_start + (i * window_size_sec)
        win_end = win_start + window_size_sec
        
        window_features = {
            'subject_id': subject_id,
            'session': exam_type,
            'window_id': i,
            'timestamp_start': win_start
        }
        
        valid_window = True
        for key, sig_info in signals.items():
            mask = (sig_info['ts'] >= win_start) & (sig_info['ts'] < win_end)
            win_data = sig_info['data'][mask]
            
            if len(win_data) == 0:
                valid_window = False
                break
                
            if key == 'acc':
                features = extract_acc_features(win_data, sig_info['freq'])
            else:
                features = extract_window_features(win_data, key, sig_info['freq'])
                
            window_features.update(features)
            
        if valid_window:
            windows_data.append(window_features)
            
    df = pd.DataFrame(windows_data)
    logger.info(f"   Generated {len(df)} windows (length={window_size_sec}s)")
    return df

def main():
    base_dir = Path(__file__).parent.parent / 'raw_data_stress'
    if not base_dir.exists():
        logger.error(f"Cannot find dataset path: {base_dir}")
        return
        
    subjects = [f'S{i}' for i in range(1, 11)]
    exams = ['Midterm 1', 'Midterm 2', 'Final']
    
    all_dfs = []
    
    for sub in subjects:
        sub_dir = base_dir / sub
        if not sub_dir.exists():
            continue
            
        for exam in exams:
            exam_dir = sub_dir / exam
            if exam_dir.exists():
                df = process_session(exam_dir, sub, exam, window_size_sec=60)
                if not df.empty:
                    all_dfs.append(df)
                    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        out_path = Path(__file__).parent.parent / 'processed_data' / 'stress_features_v1.csv'
        out_path.parent.mkdir(exist_ok=True)
        final_df.to_csv(out_path, index=False)
        logger.info(f"\nSaved integrated stress dataset! Shape: {final_df.shape}")
        logger.info(f"Path: {out_path}")
    else:
        logger.warning("No data extracted.")

if __name__ == "__main__":
    main()
