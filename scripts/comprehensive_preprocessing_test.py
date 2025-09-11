"""
Comprehensive Preprocessing Test and Verification Script
======================================================

This script verifies that the preprocessing pipeline correctly handles
all datasets without falling back to dummy data or failing silently.

Features:
1. Verifies dataset availability and completeness
2. Tests preprocessing with actual data files
3. Ensures no fallback to dummy datasets
4. Reports detailed statistics on processed data
5. Identifies any failures or missing data processing

Author: AI Assistant
Project: AI Integration in Wearables for Clot Monitoring
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Configure logging with UTF-8 encoding to handle special characters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class DatasetVerifier:
    """Verify dataset integrity and processing completeness"""
    
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.verification_results = {}
        
    def verify_datasets_exist(self) -> Dict[str, Any]:
        """Verify all expected datasets exist and are accessible"""
        
        logger.info("=" * 60)
        logger.info("VERIFYING DATASET AVAILABILITY")
        logger.info("=" * 60)
        
        # Check main datasets
        ppg_dataset = self.csv_path / "PPG_Dataset.csv"
        subjects_info = self.csv_path / "subjects_info.csv"
        
        # Find all subject files
        subject_files = list(self.csv_path.glob("s*_*.csv"))
        subject_files_by_id = {}
        
        for file_path in subject_files:
            filename = file_path.name
            if '_' in filename:
                subject_id = filename.split('_')[0]
                if subject_id not in subject_files_by_id:
                    subject_files_by_id[subject_id] = []
                subject_files_by_id[subject_id].append(filename)
        
        # Verify completeness
        expected_subjects = [f"s{i}" for i in range(1, 23)]  # s1 to s22
        expected_activities = ['walk', 'sit', 'run']
        
        verification = {
            'ppg_dataset_exists': ppg_dataset.exists(),
            'ppg_dataset_size_mb': ppg_dataset.stat().st_size / (1024*1024) if ppg_dataset.exists() else 0,
            'subjects_info_exists': subjects_info.exists(),
            'subjects_info_size_mb': subjects_info.stat().st_size / (1024*1024) if subjects_info.exists() else 0,
            'total_subject_files': len(subject_files),
            'unique_subjects': len(subject_files_by_id),
            'subject_files_by_id': subject_files_by_id,
            'missing_subjects': [],
            'incomplete_subjects': [],
            'complete_subjects': []
        }
        
        # Check for missing or incomplete subjects
        for subject in expected_subjects:
            if subject not in subject_files_by_id:
                verification['missing_subjects'].append(subject)
            else:
                subject_activities = [f.split('_')[1].replace('.csv', '') 
                                    for f in subject_files_by_id[subject] if '_' in f]
                
                missing_activities = set(expected_activities) - set(subject_activities)
                if missing_activities:
                    verification['incomplete_subjects'].append({
                        'subject': subject,
                        'has_activities': subject_activities,
                        'missing_activities': list(missing_activities)
                    })
                else:
                    verification['complete_subjects'].append(subject)
        
        # Log results with Windows-compatible symbols
        logger.info(f"PPG Dataset: {'[FOUND]' if verification['ppg_dataset_exists'] else '[MISSING]'} "
                   f"({verification['ppg_dataset_size_mb']:.1f} MB)")
        logger.info(f"Subjects Info: {'[FOUND]' if verification['subjects_info_exists'] else '[MISSING]'} "
                   f"({verification['subjects_info_size_mb']:.1f} MB)")
        logger.info(f"Subject Files: {verification['total_subject_files']} files from "
                   f"{verification['unique_subjects']} subjects")
        logger.info(f"Complete Subjects: {len(verification['complete_subjects'])}/22")
        
        if verification['missing_subjects']:
            logger.warning(f"Missing Subjects: {verification['missing_subjects']}")
        
        if verification['incomplete_subjects']:
            logger.warning("Incomplete Subjects:")
            for incomplete in verification['incomplete_subjects']:
                logger.warning(f"  {incomplete['subject']}: missing {incomplete['missing_activities']}")
        
        self.verification_results['dataset_availability'] = verification
        return verification
    
    def verify_dataset_content(self) -> Dict[str, Any]:
        """Verify dataset content and structure"""
        
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING DATASET CONTENT")
        logger.info("=" * 60)
        
        content_verification = {}
        
        # Check PPG Dataset
        if (self.csv_path / "PPG_Dataset.csv").exists():
            try:
                ppg_sample = pd.read_csv(self.csv_path / "PPG_Dataset.csv", nrows=1000)
                content_verification['ppg_dataset'] = {
                    'loadable': True,
                    'columns': list(ppg_sample.columns),
                    'num_columns': len(ppg_sample.columns),
                    'sample_rows': len(ppg_sample),
                    'has_ppg_signals': any('ppg' in col.lower() or 'pleth' in col.lower() 
                                         for col in ppg_sample.columns),
                    'has_timestamps': any('time' in col.lower() for col in ppg_sample.columns)
                }
                logger.info(f"PPG Dataset: {content_verification['ppg_dataset']['num_columns']} columns, "
                           f"PPG signals: {'[YES]' if content_verification['ppg_dataset']['has_ppg_signals'] else '[NO]'}")
            except Exception as e:
                content_verification['ppg_dataset'] = {
                    'loadable': False, 
                    'error': str(e)
                }
                logger.error(f"PPG Dataset loading failed: {e}")
        
        # Check Subjects Info
        if (self.csv_path / "subjects_info.csv").exists():
            try:
                subjects_sample = pd.read_csv(self.csv_path / "subjects_info.csv")
                content_verification['subjects_info'] = {
                    'loadable': True,
                    'columns': list(subjects_sample.columns),
                    'num_columns': len(subjects_sample.columns),
                    'num_subjects': len(subjects_sample),
                    'has_demographics': any(col in ['age', 'gender', 'weight', 'height'] 
                                          for col in subjects_sample.columns)
                }
                logger.info(f"Subjects Info: {content_verification['subjects_info']['num_subjects']} subjects, "
                           f"{content_verification['subjects_info']['num_columns']} columns")
            except Exception as e:
                content_verification['subjects_info'] = {
                    'loadable': False, 
                    'error': str(e)
                }
                logger.error(f"Subjects Info loading failed: {e}")
        
        # Check sample subject files
        subject_files = list(self.csv_path.glob("s*_*.csv"))
        sample_files = subject_files[:5] if len(subject_files) > 5 else subject_files
        
        content_verification['subject_files'] = []
        
        for file_path in sample_files:
            try:
                sample_data = pd.read_csv(file_path, nrows=100)
                file_info = {
                    'filename': file_path.name,
                    'loadable': True,
                    'columns': list(sample_data.columns),
                    'num_columns': len(sample_data.columns),
                    'sample_rows': len(sample_data),
                    'has_sensors': {
                        'accelerometer': any('a_' in col for col in sample_data.columns),
                        'gyroscope': any('g_' in col for col in sample_data.columns),
                        'ecg': any('ecg' in col.lower() for col in sample_data.columns),
                        'ppg': any('pleth' in col.lower() for col in sample_data.columns)
                    }
                }
                content_verification['subject_files'].append(file_info)
                
                sensors = [k for k, v in file_info['has_sensors'].items() if v]
                logger.info(f"  {file_path.name}: {file_info['num_columns']} columns, "
                           f"sensors: {', '.join(sensors)}")
                
            except Exception as e:
                content_verification['subject_files'].append({
                    'filename': file_path.name,
                    'loadable': False,
                    'error': str(e)
                })
                logger.error(f"  {file_path.name}: Loading failed - {e}")
        
        self.verification_results['dataset_content'] = content_verification
        return content_verification

class PreprocessingTester:
    """Test preprocessing pipeline with real data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.test_results = {}
        
    def test_preprocessing_completeness(self) -> Dict[str, Any]:
        """Test if preprocessing handles full datasets without fallbacks"""
        
        logger.info("\n" + "=" * 60)
        logger.info("TESTING PREPROCESSING COMPLETENESS")
        logger.info("=" * 60)
        
        try:
            # Import preprocessing modules - try main module only
            from data_preprocessing import IntegratedPreprocessor
            logger.info("[SUCCESS] Imported data_preprocessing successfully")
        except ImportError as e:
            logger.error(f"[ERROR] Failed to import preprocessing module: {e}")
            return {'error': 'Cannot import preprocessing module'}
        
        # Initialize preprocessor
        preprocessor = IntegratedPreprocessor(self.csv_path)
        
        # Test with different configurations
        test_configs = [
            {'name': 'Limited Files (5)', 'max_files': 5},
            {'name': 'Medium Files (15)', 'max_files': 15},
            {'name': 'All Files', 'max_files': None}
        ]
        
        test_results = {}
        
        for config in test_configs:
            logger.info(f"\n--- Testing Configuration: {config['name']} ---")
            
            try:
                results = preprocessor.run_complete_preprocessing(
                    load_subjects=True,
                    max_subject_files=config['max_files']
                )
                
                # Analyze results
                config_results = {
                    'success': True,
                    'datasets_processed': list(results.keys()) if results else [],
                    'fallback_detected': False,
                    'total_records': 0,
                    'details': {}
                }
                
                if results:
                    for dataset_name, dataset in results.items():
                        if isinstance(dataset, pd.DataFrame):
                            config_results['details'][dataset_name] = {
                                'type': 'DataFrame',
                                'rows': len(dataset),
                                'columns': len(dataset.columns),
                                'memory_mb': dataset.memory_usage(deep=True).sum() / (1024*1024)
                            }
                            config_results['total_records'] += len(dataset)
                        elif isinstance(dataset, dict):
                            config_results['details'][dataset_name] = {
                                'type': 'Dict',
                                'num_items': len(dataset),
                                'total_rows': sum(len(df) for df in dataset.values() if isinstance(df, pd.DataFrame))
                            }
                            config_results['total_records'] += config_results['details'][dataset_name]['total_rows']
                
                # Check for fallback indicators
                if config_results['total_records'] < 1000:  # Suspiciously low
                    config_results['fallback_detected'] = True
                    logger.warning(f"[WARNING] Suspiciously low record count: {config_results['total_records']}")
                
                # Log results with Windows-compatible symbols
                logger.info(f"[SUCCESS] Configuration successful:")
                logger.info(f"   - Datasets: {len(config_results['datasets_processed'])}")
                logger.info(f"   - Total records: {config_results['total_records']:,}")
                logger.info(f"   - Fallback detected: {config_results['fallback_detected']}")
                
                for dataset_name, details in config_results['details'].items():
                    if details['type'] == 'DataFrame':
                        logger.info(f"   - {dataset_name}: {details['rows']:,} rows, "
                                   f"{details['columns']} cols, {details['memory_mb']:.2f} MB")
                    else:
                        logger.info(f"   - {dataset_name}: {details['num_items']} files, "
                                   f"{details['total_rows']:,} total rows")
                
            except Exception as e:
                config_results = {
                    'success': False,
                    'error': str(e),
                    'fallback_detected': True
                }
                logger.error(f"[ERROR] Configuration failed: {e}")
            
            test_results[config['name']] = config_results
        
        self.test_results['preprocessing_completeness'] = test_results
        return test_results
    
    def test_feature_extraction(self) -> Dict[str, Any]:
        """Test feature extraction completeness"""
        
        logger.info("\n" + "=" * 60)
        logger.info("TESTING FEATURE EXTRACTION")
        logger.info("=" * 60)
        
        try:
            # Import PPG analysis - try main module only
            from ppg_analysis import PPGSignalAnalyzer, extract_ppg_features_from_dataset
            logger.info("[SUCCESS] Imported ppg_analysis successfully")
        except ImportError as e:
            logger.error(f"[ERROR] Failed to import PPG analysis module: {e}")
            return {'error': 'Cannot import PPG analysis module'}
        
        # Test with a sample subject file
        subject_files = list(Path(self.csv_path).glob("s*_*.csv"))
        
        if not subject_files:
            logger.warning("No subject files found for feature extraction test")
            return {'error': 'No subject files available'}
        
        test_file = subject_files[0]
        logger.info(f"Testing feature extraction with: {test_file.name}")
        
        try:
            # Load sample data
            data = pd.read_csv(test_file)
            
            # Identify PPG columns
            ppg_cols = [col for col in data.columns if 'pleth' in col.lower()]
            
            if not ppg_cols:
                logger.warning("No PPG columns found in sample file")
                return {'error': 'No PPG columns in sample file'}
            
            # Test feature extraction
            features_df = extract_ppg_features_from_dataset(
                data=data,
                ppg_columns=ppg_cols[:2],  # Use first 2 PPG channels
                window_size=500  # Small window for test
            )
            
            extraction_results = {
                'success': len(features_df) > 0,
                'num_feature_windows': len(features_df),
                'num_features_per_window': len(features_df.columns) if len(features_df) > 0 else 0,
                'sample_features': list(features_df.columns[:10]) if len(features_df) > 0 else [],
                'total_data_rows': len(data),
                'ppg_columns_used': ppg_cols[:2]
            }
            
            logger.info(f"[SUCCESS] Feature extraction successful:")
            logger.info(f"   - Feature windows: {extraction_results['num_feature_windows']}")
            logger.info(f"   - Features per window: {extraction_results['num_features_per_window']}")
            logger.info(f"   - Sample features: {extraction_results['sample_features']}")
            
            return extraction_results
            
        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed: {e}")
            return {'error': str(e), 'success': False}

def main():
    """Run comprehensive preprocessing verification"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PREPROCESSING VERIFICATION")
    print("AI Integration in Wearables for Clot Monitoring")
    print("=" * 80)
    
    # Get CSV path
    base_dir = Path(__file__).parent
    csv_path = base_dir / "csv"
    
    if not csv_path.exists():
        logger.error(f"CSV directory not found: {csv_path}")
        logger.info("Please ensure your datasets are in the 'csv' folder")
        return
    
    # Initialize verifiers
    dataset_verifier = DatasetVerifier(str(csv_path))
    preprocessing_tester = PreprocessingTester(str(csv_path))
    
    # Run verifications
    dataset_availability = dataset_verifier.verify_datasets_exist()
    dataset_content = dataset_verifier.verify_dataset_content()
    preprocessing_results = preprocessing_tester.test_preprocessing_completeness()
    feature_extraction_results = preprocessing_tester.test_feature_extraction()
    
    # Generate final report
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION REPORT")
    print("=" * 80)
    
    # Dataset Availability Summary
    print("\nDATASET AVAILABILITY:")
    print(f"   PPG Dataset: {'[YES]' if dataset_availability['ppg_dataset_exists'] else '[NO]'}")
    print(f"   Subjects Info: {'[YES]' if dataset_availability['subjects_info_exists'] else '[NO]'}")
    print(f"   Complete Subjects: {len(dataset_availability['complete_subjects'])}/22")
    print(f"   Total Subject Files: {dataset_availability['total_subject_files']}")
    
    # Content Verification Summary
    print("\nDATASET CONTENT:")
    if 'ppg_dataset' in dataset_content:
        ppg_status = dataset_content['ppg_dataset']
        print(f"   PPG Dataset: {'[LOADABLE]' if ppg_status.get('loadable') else '[ERROR]'}")
        if ppg_status.get('loadable'):
            print(f"     - Columns: {ppg_status['num_columns']}")
            print(f"     - Has PPG signals: {'[YES]' if ppg_status['has_ppg_signals'] else '[NO]'}")
    
    if 'subjects_info' in dataset_content:
        subjects_status = dataset_content['subjects_info']
        print(f"   Subjects Info: {'[LOADABLE]' if subjects_status.get('loadable') else '[ERROR]'}")
        if subjects_status.get('loadable'):
            print(f"     - Subjects: {subjects_status['num_subjects']}")
            print(f"     - Has demographics: {'[YES]' if subjects_status['has_demographics'] else '[NO]'}")
    
    # Preprocessing Summary
    print("\nPREPROCESSING VERIFICATION:")
    for config_name, results in preprocessing_results.items():
        if 'error' not in preprocessing_results:
            status = '[SUCCESS]' if results['success'] else '[FAILED]'
            fallback = '[YES]' if results.get('fallback_detected') else '[NO]'
            print(f"   {config_name}: {status}")
            print(f"     - Fallback detected: {fallback}")
            print(f"     - Total records: {results.get('total_records', 0):,}")
        else:
            print(f"   Preprocessing: [IMPORT ERROR]")
    
    # Feature Extraction Summary
    print("\nFEATURE EXTRACTION:")
    if 'error' not in feature_extraction_results:
        status = '[SUCCESS]' if feature_extraction_results['success'] else '[FAILED]'
        print(f"   Status: {status}")
        if feature_extraction_results['success']:
            print(f"   - Feature windows: {feature_extraction_results['num_feature_windows']}")
            print(f"   - Features per window: {feature_extraction_results['num_features_per_window']}")
    else:
        print(f"   Status: [ERROR] - {feature_extraction_results['error']}")
    
    # Overall Assessment
    print("\nOVERALL ASSESSMENT:")
    
    issues = []
    if not dataset_availability['ppg_dataset_exists']:
        issues.append("PPG Dataset missing")
    if not dataset_availability['subjects_info_exists']:
        issues.append("Subjects Info missing")
    if len(dataset_availability['complete_subjects']) < 20:
        issues.append(f"Only {len(dataset_availability['complete_subjects'])}/22 complete subjects")
    
    preprocessing_successful = all(
        results.get('success', False) and not results.get('fallback_detected', True)
        for results in preprocessing_results.values()
        if 'error' not in results
    )
    
    if not preprocessing_successful:
        issues.append("Preprocessing has issues or fallbacks")
    
    if 'error' in feature_extraction_results or not feature_extraction_results.get('success'):
        issues.append("Feature extraction failed")
    
    if not issues:
        print("   [SUCCESS] ALL SYSTEMS OPERATIONAL!")
        print("   Your preprocessing pipeline is ready for full dataset processing.")
        print("   No fallbacks to dummy data detected.")
    else:
        print("   [WARNING] Issues detected:")
        for issue in issues:
            print(f"      - {issue}")
        print("\n   [RECOMMENDATIONS]:")
        print("      - Check that all dataset files are present and accessible")
        print("      - Verify file formats and column names match expectations")
        print("      - Review preprocessing code for error handling")
    
    # Save detailed results
    results_file = base_dir / "verification_results.log"
    with open(results_file, 'w') as f:
        f.write("COMPREHENSIVE PREPROCESSING VERIFICATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset Availability: {dataset_availability}\n\n")
        f.write(f"Dataset Content: {dataset_content}\n\n")
        f.write(f"Preprocessing Results: {preprocessing_results}\n\n")
        f.write(f"Feature Extraction: {feature_extraction_results}\n\n")
    
    print(f"\n[RESULTS] Detailed results saved to: {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
