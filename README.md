# AI Integration in Wearables for Clot Monitoring
## Data Preprocessing Pipeline

This project focuses on using AI to detect early signs of blood clots (like DVT) from wearable sensor data, emphasizing machine learning, data processing, and anomaly detection for cardiovascular health monitoring.

## 📁 Project Structure

```
├── csv/                          # Raw CSV datasets
│   ├── PPG_Dataset.csv          # Large PPG dataset
│   ├── subjects_info.csv        # Subject demographics and physiological data
│   ├── s1_walk.csv, s1_run.csv  # Individual subject activity data
│   └── s2_sit.csv, ...          # (s1-s22 with walk/run/sit activities)
├── data_preprocessing.py         # Main preprocessing pipeline
├── ppg_analysis.py              # Specialized PPG signal analysis
├── run_preprocessing.py          # Quick start script
├── requirements.txt             # Python dependencies
├── processed_data/              # Output directory (created automatically)
│   ├── features/                # Extracted features
│   ├── reports/                 # Processing reports
│   └── visualizations/          # Generated plots
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing

```bash
python run_preprocessing.py
```

This will:
- ✅ Load and preprocess all available datasets
- ✅ Extract comprehensive features for ML training
- ✅ Create integrated datasets with risk indicators
- ✅ Generate quality reports and summaries
- ✅ Save everything to the `processed_data/` folder

## 📊 Datasets Overview

### 1. **PPG_Dataset.csv**
- Large photoplethysmography dataset
- Contains PPG signals for cardiac analysis
- Used for heart rate variability and blood flow pattern analysis

### 2. **Subject Data (s1-s22)**
- Multi-sensor physiological data from 22 subjects
- Three activities per subject: walking, running, sitting
- Sensors include:
  - **ECG**: Electrocardiogram signals
  - **PPG**: Multiple photoplethysmography channels (pleth_1 to pleth_6)
  - **Temperature**: Multiple temperature sensors
  - **Accelerometer**: Motion data (a_x, a_y, a_z)
  - **Gyroscope**: Rotation data (g_x, g_y, g_z)
  - **LC sensors**: Additional physiological measurements

### 3. **subjects_info.csv**
- Demographics: age, gender, height, weight
- Physiological baselines: blood pressure, heart rate, SpO2
- Pre/post activity measurements for change analysis

## 🔬 Feature Engineering

The preprocessing pipeline extracts multiple categories of features:

### **Physiological Features**
- Heart rate variability (HRV) metrics
- Pulse transit time (PTT) for blood pressure estimation
- Signal quality indicators
- Perfusion index measurements

### **Motion Features**
- Accelerometer and gyroscope magnitude
- Activity level indicators
- Motion artifact detection

### **Risk Indicators**
- Blood pressure change patterns
- Heart rate variability anomalies
- Age and BMI risk factors
- Composite risk scoring for clot monitoring

### **Signal Processing Features**
- Statistical moments (mean, std, skew, kurtosis)
- Frequency domain characteristics
- Time-domain variability measures
- Peak detection and morphology analysis

## 🏥 Clot Monitoring Focus

The preprocessing specifically targets features relevant for blood clot detection:

### **Key Indicators**
1. **Pulse Transit Time (PTT)**: Changes can indicate blood pressure variations
2. **Heart Rate Variability**: Alterations may suggest circulatory issues
3. **PPG Signal Quality**: Reduced perfusion can indicate clot formation
4. **Motion Patterns**: Sedentary behavior increases clot risk
5. **Demographic Risk Factors**: Age, BMI, and medical history

### **Risk Assessment**
- Automated anomaly detection in PPG signals
- Baseline comparison for individual monitoring
- Multi-modal sensor fusion for comprehensive assessment
- Real-time risk scoring algorithms

## 📈 Outputs

After preprocessing, you'll have:

### **Ready-to-use datasets:**
- `subjects_info.csv` - Processed demographic data
- `subject_features.csv` - Extracted features from sensor data
- `integrated_features.csv` - Combined features with risk indicators
- `advanced_ppg_features.csv` - Specialized PPG analysis results

### **Analysis reports:**
- `preprocessing_summary.csv` - Dataset statistics
- `data_quality_summary.csv` - Quality metrics
- `preprocessing_report.txt` - Detailed processing report

## 🛠️ Advanced Usage

### Custom Processing

```python
from data_preprocessing import IntegratedPreprocessor

# Initialize preprocessor
preprocessor = IntegratedPreprocessor("path/to/csv/folder")

# Custom processing
results = preprocessor.run_complete_preprocessing(
    load_ppg=True,           # Process large PPG dataset
    load_subjects=True,      # Process subject files
    max_subject_files=22     # Process all subjects
)
```

### PPG Signal Analysis

```python
from ppg_analysis import PPGSignalAnalyzer

# Initialize analyzer
analyzer = PPGSignalAnalyzer(sampling_rate=500)

# Analyze PPG signal
results = analyzer.analyze_ppg_window(ppg_data, ecg_data)

# Check for anomalies
anomalies = analyzer.detect_blood_flow_anomalies(ppg_data, baseline_metrics)
```

## 🔍 Data Quality Features

- **Missing value handling**: Multiple imputation strategies
- **Outlier detection**: IQR and Z-score methods
- **Signal filtering**: Butterworth bandpass filters
- **Quality assessment**: SNR, correlation, and artifact detection
- **Normalization**: Standard, robust, and min-max scaling options

## 📋 Next Steps for ML Development

1. **Load processed features**: Use `integrated_features.csv`
2. **Split data**: Train/validation/test sets by subject
3. **Model selection**: Try classification (clot risk) or regression (risk scores)
4. **Feature selection**: Use domain knowledge and statistical methods
5. **Validation**: Cross-validation with subject-wise splits
6. **Evaluation**: Focus on medical relevance and interpretability

## 🔧 Troubleshooting

### Common Issues:

**Large PPG dataset**: 
- The script automatically skips files >100MB in quick mode
- Set `load_ppg=True` for full processing if needed

**Memory issues**: 
- Reduce `max_subject_files` parameter
- Process in smaller chunks
- Use `chunk_size` parameter for large files

**Missing dependencies**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📝 Logging

Processing logs are saved to `preprocessing.log` for detailed debugging and monitoring.

## 🎯 Project Goals

This preprocessing pipeline prepares data for developing AI models that can:
- Monitor cardiovascular health through wearable sensors
- Detect early signs of blood clot formation
- Provide real-time risk assessment
- Enable preventive healthcare interventions
- Support clinical decision-making with data-driven insights

---

**Ready to build your clot monitoring AI system!** 🏥💓🤖
