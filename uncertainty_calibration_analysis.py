"""
Uncertainty Calibration Analysis for Blood Clot Risk Prediction
================================================================

Purpose: Validate that our uncertainty estimates correlate with actual prediction errors.
Expected: High uncertainty should correlate with lower accuracy.

Example: If model says 50% uncertain, it should be wrong ~50% of the time.
Date: 2025-10-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_latest_models(models_dir='trained_models'):
    """Load the most recently trained models"""
    import glob
    import os

    metadata_files = glob.glob(os.path.join(models_dir, 'model_metadata_*.pkl'))
    if not metadata_files:
        raise FileNotFoundError(f"No model metadata found in {models_dir}")

    latest_metadata_file = max(metadata_files, key=os.path.getctime)
    metadata = joblib.load(latest_metadata_file)
    timestamp = '_'.join(os.path.basename(latest_metadata_file).split('_')[2:]).replace('.pkl', '')

    models = {
        'classifier': joblib.load(os.path.join(models_dir, f'best_classifier_{metadata["best_classifier"]}_{timestamp}.pkl')),
        'regressor': joblib.load(os.path.join(models_dir, f'best_regressor_{metadata["best_regressor"]}_{timestamp}.pkl')),
        'scaler': joblib.load(os.path.join(models_dir, f'scaler_{timestamp}.pkl')),
        'label_encoder': joblib.load(os.path.join(models_dir, f'label_encoder_{timestamp}.pkl')),
        'metadata': metadata
    }

    print(f"Loaded models from {timestamp}")
    print(f"Classifier: {metadata['best_classifier']}")
    print(f"Regressor: {metadata['best_regressor']}\n")

    return models

def calculate_uncertainty(probabilities):
    """Calculate Shannon entropy as uncertainty measure"""
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    entropy = -np.sum(probabilities * np.log2(probabilities), axis=1)
    max_entropy = np.log2(probabilities.shape[1])
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def predict_with_uncertainty(models, patient_data):
    """Generate predictions with uncertainty for all patients"""
    feature_columns = models['metadata']['feature_columns']
    X = patient_data[feature_columns]
    X_scaled_array = models['scaler'].transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_columns, index=X.index)

    results = []

    for idx in range(len(X_scaled)):
        X_sample = X_scaled.iloc[[idx]]

        # Get predictions from both models
        class_probs = models['classifier'].predict_proba(X_sample)[0]
        predicted_class_idx = np.argmax(class_probs)
        predicted_category = models['label_encoder'].inverse_transform([predicted_class_idx])[0]
        confidence = class_probs[predicted_class_idx]

        # Calculate uncertainty
        epsilon = 1e-10
        entropy = -np.sum(class_probs * np.log(class_probs + epsilon))
        max_entropy = np.log(len(class_probs))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0

        # Get actual category
        actual_category = patient_data.loc[patient_data.index[idx], 'risk_category']

        # Check if prediction is correct
        is_correct = (predicted_category == actual_category)

        results.append({
            'patient_id': patient_data.index[idx],
            'predicted_category': predicted_category,
            'actual_category': actual_category,
            'uncertainty': uncertainty * 100,  # Convert to percentage
            'confidence': confidence * 100,
            'correct': is_correct
        })

    return pd.DataFrame(results)

def main():
    print("="*80)
    print("UNCERTAINTY CALIBRATION ANALYSIS")
    print("="*80)
    print("\nGoal: Validate that uncertainty correlates with prediction errors")
    print("Expected: High uncertainty -> Lower accuracy\n")

    # Load models
    models = load_latest_models()

    # Load data (same 600 patients as pilot study)
    print("Loading patient data...")
    data = pd.read_csv('processed_data/integrated_features_improved_balanced.csv')
    test_patients = data.sample(n=600, random_state=42)
    print(f"Testing on {len(test_patients)} patients\n")

    # Generate predictions with uncertainty
    print("Generating predictions with uncertainty...")
    results_df = predict_with_uncertainty(models, test_patients)

    print(f"\nTotal predictions: {len(results_df)}")
    print(f"Overall accuracy: {results_df['correct'].mean() * 100:.2f}%")
    print(f"Average uncertainty: {results_df['uncertainty'].mean():.2f}%")

    # Calibration Analysis: Bin by uncertainty ranges
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS: Uncertainty vs Actual Accuracy")
    print("="*80)

    # Define uncertainty bins
    bins = [0, 20, 40, 60, 80, 100]
    bin_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']

    results_df['uncertainty_bin'] = pd.cut(results_df['uncertainty'],
                                            bins=bins,
                                            labels=bin_labels,
                                            include_lowest=True)

    # Calculate accuracy per bin
    calibration_stats = []

    print("\n" + "-"*80)
    print(f"{'Uncertainty':^17} | {'Count':^7} | {'Actual Accuracy':^17} | {'Expected Accuracy':^19} | {'Calibration':^14}")
    print("-"*80)

    for bin_label in bin_labels:
        bin_data = results_df[results_df['uncertainty_bin'] == bin_label]

        if len(bin_data) > 0:
            actual_accuracy = bin_data['correct'].mean() * 100

            # Expected accuracy: inverse of uncertainty midpoint
            # e.g., 30% uncertainty → ~70% expected accuracy
            bin_start = float(bin_label.split('-')[0])
            bin_end = float(bin_label.split('-')[1].replace('%', ''))
            bin_midpoint = (bin_start + bin_end) / 2
            expected_accuracy = 100 - bin_midpoint

            # Calibration error: how far off we are
            calibration_error = abs(actual_accuracy - expected_accuracy)

            # Determine calibration status
            if calibration_error < 5:
                status = "Excellent"
            elif calibration_error < 10:
                status = "Good"
            elif calibration_error < 15:
                status = "Fair"
            else:
                status = "Poor"

            calibration_stats.append({
                'bin': bin_label,
                'count': len(bin_data),
                'actual_accuracy': actual_accuracy,
                'expected_accuracy': expected_accuracy,
                'calibration_error': calibration_error,
                'status': status
            })

            print(f"{bin_label:^17} | {len(bin_data):^7} | {actual_accuracy:^15.2f}% | {expected_accuracy:^17.1f}% | {status:^14}")
        else:
            print(f"{bin_label:^17} | {0:^7} | {'N/A':^17} | {'N/A':^19} | {'No data':^14}")

    print("-"*80)

    calibration_df = pd.DataFrame(calibration_stats)

    # Calculate mean calibration error
    if len(calibration_df) > 0:
        mean_calibration_error = calibration_df['calibration_error'].mean()
        print(f"\nMean Calibration Error: {mean_calibration_error:.2f}%")

        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)
        if mean_calibration_error < 5:
            print("EXCELLENT: Uncertainty is extremely well-calibrated with actual accuracy")
            print("   -> Model's uncertainty estimates are highly trustworthy")
        elif mean_calibration_error < 10:
            print("GOOD: Uncertainty is well-calibrated with actual accuracy")
            print("   -> Model's uncertainty estimates are reliable")
        elif mean_calibration_error < 15:
            print("FAIR: Uncertainty is reasonably calibrated")
            print("   -> Minor adjustments may improve calibration")
        else:
            print("NEEDS IMPROVEMENT: Uncertainty calibration needs adjustment")
            print("   -> Consider recalibrating uncertainty thresholds")

        # Generate calibration plot
        print("\nGenerating calibration plots...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Calibration curve
        ax1 = axes[0]
        if len(calibration_df) > 0:
            x_values = [(float(row['bin'].split('-')[0]) + float(row['bin'].split('-')[1].replace('%', ''))) / 2
                        for _, row in calibration_df.iterrows()]

            ax1.plot(x_values, calibration_df['actual_accuracy'],
                     marker='o', markersize=10, linewidth=2.5,
                     label='Actual Accuracy', color='#2563eb')
            ax1.plot(x_values, calibration_df['expected_accuracy'],
                     marker='s', markersize=10, linewidth=2.5, linestyle='--',
                     label='Expected Accuracy (Perfect Calibration)', color='#dc2626')

            # Add diagonal reference line
            ax1.plot([0, 100], [100, 0], 'k--', alpha=0.3, linewidth=1, label='Perfect Inverse')

            ax1.set_xlabel('Uncertainty (Midpoint of Range %)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Uncertainty Calibration Curve\n(Closer = Better Calibrated)',
                          fontsize=13, fontweight='bold', pad=15)
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 105)
            ax1.set_xlim(0, 100)

        # Plot 2: Sample distribution by uncertainty
        ax2 = axes[1]
        bin_counts = results_df['uncertainty_bin'].value_counts().sort_index()
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#7c3aed']
        bars = ax2.bar(range(len(bin_counts)), bin_counts.values,
                       color=colors[:len(bin_counts)], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Uncertainty Range', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Distribution by Uncertainty\n(Shows model confidence)',
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_xticks(range(len(bin_counts)))
        ax2.set_xticklabels(bin_counts.index, rotation=0)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, v in enumerate(bin_counts.values):
            ax2.text(i, v + max(bin_counts.values)*0.02, str(v),
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.tight_layout()

        # Save plot
        Path('pilot_study_plots').mkdir(exist_ok=True)
        plt.savefig('pilot_study_plots/7_uncertainty_calibration.png', dpi=300, bbox_inches='tight')
        print("Saved: pilot_study_plots/7_uncertainty_calibration.png")

        # Save detailed results
        results_df.to_csv('pilot_study_plots/uncertainty_calibration_results.csv', index=False)
        print("Saved: pilot_study_plots/uncertainty_calibration_results.csv")

        # Additional Analysis: High-risk predictions
        print("\n" + "="*80)
        print("HIGH-RISK PREDICTION ANALYSIS")
        print("="*80)

        high_risk_predictions = results_df[results_df['predicted_category'].isin(['Critical', 'High'])]
        print(f"\nTotal High-Risk predictions: {len(high_risk_predictions)}")
        if len(high_risk_predictions) > 0:
            print(f"Accuracy on High-Risk predictions: {high_risk_predictions['correct'].mean() * 100:.2f}%")
            print(f"Average uncertainty on High-Risk predictions: {high_risk_predictions['uncertainty'].mean():.2f}%")
            print(f"Average confidence on High-Risk predictions: {high_risk_predictions['confidence'].mean():.2f}%")

        # Uncertainty threshold validation
        print("\n" + "="*80)
        print("UNCERTAINTY THRESHOLD VALIDATION")
        print("="*80)

        # Current threshold: 55%
        high_uncertainty = results_df[results_df['uncertainty'] > 55]
        print(f"\nCurrent threshold: 55% uncertainty")
        print(f"Predictions above threshold: {len(high_uncertainty)} ({len(high_uncertainty)/len(results_df)*100:.1f}%)")
        if len(high_uncertainty) > 0:
            print(f"Accuracy for high uncertainty cases: {high_uncertainty['correct'].mean() * 100:.2f}%")
            print(f"Error rate: {(1 - high_uncertainty['correct'].mean()) * 100:.2f}%")

        print("\n" + "="*80)
        print("CALIBRATION ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   - Mean Calibration Error: {mean_calibration_error:.2f}%")
        print(f"   - Overall Accuracy: {results_df['correct'].mean() * 100:.2f}%")
        print(f"   - Total Predictions: {len(results_df)}")
        print(f"   - Calibration Status: {calibration_stats[0]['status'] if calibration_stats else 'N/A'}")
        print(f"\nResults saved to pilot_study_plots/")

if __name__ == "__main__":
    main()
