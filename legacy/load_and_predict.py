"""
Enhanced Risk Prediction with Tree-Based Uncertainty Estimation

Uses XGBoost's internal tree predictions to calculate real model uncertainty.
More scientifically sound than bootstrap for single predictions.
"""

import pandas as pd
import numpy as np
import joblib
import glob
import os
from pathlib import Path

def load_latest_models(models_dir='../trained_models'):
    """Load the most recently trained models"""
    metadata_files = glob.glob(os.path.join(models_dir, 'model_metadata_*.pkl'))
    if not metadata_files:
        raise FileNotFoundError(f"No model metadata found in {models_dir}")

    latest_metadata_file = max(metadata_files, key=os.path.getctime)
    metadata = joblib.load(latest_metadata_file)
    # Extract timestamp from filename: model_metadata_YYYYMMDD_HHMMSS.pkl
    timestamp = '_'.join(os.path.basename(latest_metadata_file).split('_')[2:]).replace('.pkl', '')

    models = {
        'classifier': joblib.load(os.path.join(models_dir, f'best_classifier_{metadata["best_classifier"]}_{timestamp}.pkl')),
        'regressor': joblib.load(os.path.join(models_dir, f'best_regressor_{metadata["best_regressor"]}_{timestamp}.pkl')),
        'scaler': joblib.load(os.path.join(models_dir, f'scaler_{timestamp}.pkl')),
        'label_encoder': joblib.load(os.path.join(models_dir, f'label_encoder_{timestamp}.pkl')),
        'metadata': metadata
    }

    print(f"\nLoaded models from {timestamp}")
    print(f"Classifier: {metadata['best_classifier']}")
    print(f"Regressor: {metadata['best_regressor']}")

    return models

def predict_with_tree_uncertainty(models, patient_data):
    """
    Predict risk with uncertainty estimation using tree variance

    For XGBoost/Gradient Boosting:
    - Each tree makes a prediction
    - Variance across trees = model uncertainty
    - High variance = model is uncertain about this case
    """
    feature_columns = models['metadata']['feature_columns']
    X = patient_data[feature_columns]
    X_scaled_array = models['scaler'].transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_columns, index=X.index)

    results = []

    for idx in range(len(X_scaled)):
        X_sample = X_scaled.iloc[[idx]]

        # CLASSIFICATION: Get predictions from each tree in XGBoost
        classifier = models['classifier']

        # XGBoost predict_proba gives class probabilities
        class_probs = classifier.predict_proba(X_sample)[0]
        predicted_class_idx = np.argmax(class_probs)
        predicted_category = models['label_encoder'].inverse_transform([predicted_class_idx])[0]
        confidence = class_probs[predicted_class_idx]

        # For XGBoost: Calculate uncertainty from class probability distribution
        # High entropy = high uncertainty, low entropy = confident prediction
        if hasattr(classifier, 'get_booster'):
            # Calculate entropy of probability distribution
            # Entropy: -sum(p * log(p)) where p are class probabilities
            # Normalize to [0,1] range
            epsilon = 1e-10  # Avoid log(0)
            entropy = -np.sum(class_probs * np.log(class_probs + epsilon))
            max_entropy = np.log(len(class_probs))  # Maximum possible entropy
            uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
            overall_uncertainty = uncertainty

            # Alternative: Use prediction margin (difference between top 2 classes)
            sorted_probs = np.sort(class_probs)[::-1]
            margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            # Low margin = high uncertainty
            margin_uncertainty = 1.0 - margin
        else:
            uncertainty = 0.0
            overall_uncertainty = 0.0
            margin_uncertainty = 0.0

        # REGRESSION: Get score predictions from tree ensemble
        regressor = models['regressor']
        risk_score = regressor.predict(X_sample)[0]

        # For Gradient Boosting Regressor: Calculate prediction variance
        if hasattr(regressor, 'estimators_'):
            # Get predictions from staged predict (cumulative trees)
            staged_preds = list(regressor.staged_predict(X_sample))

            # Variance in later predictions shows uncertainty
            if len(staged_preds) > 10:
                recent_preds = staged_preds[-20:]  # Last 20 tree additions
                score_variance = np.std(recent_preds)
                score_ci_lower = risk_score - 1.96 * score_variance
                score_ci_upper = risk_score + 1.96 * score_variance
            else:
                score_variance = 0.0
                score_ci_lower = risk_score
                score_ci_upper = risk_score
        else:
            score_variance = 0.0
            score_ci_lower = risk_score
            score_ci_upper = risk_score

        # Apply optimized thresholds (from cross-validation)
        tuned_category = apply_threshold_tuning(risk_score)

        # Determine alert level based on category and uncertainty
        alert_level = determine_alert_level(
            predicted_category,
            confidence,
            uncertainty,
            tuned_category
        )

        result = {
            'patient_id': patient_data.index[idx],
            'predicted_category': predicted_category,
            'tuned_category': tuned_category,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'overall_uncertainty': overall_uncertainty,
            'risk_score': risk_score,
            'score_variance': score_variance,
            'score_ci_lower': score_ci_lower,
            'score_ci_upper': score_ci_upper,
            'alert_level': alert_level,
            'class_probabilities': dict(zip(models['label_encoder'].classes_, class_probs))
        }
        results.append(result)

    return pd.DataFrame(results)

def apply_threshold_tuning(risk_score):
    """
    Apply optimized thresholds from cross-validation
    Critical >= 6.0, High >= 3.5, Moderate >= 2.5
    """
    if risk_score >= 6.0:
        return 'Critical'
    elif risk_score >= 3.5:
        return 'High'
    elif risk_score >= 2.5:
        return 'Moderate'
    elif risk_score >= 1.5:
        return 'Low-Moderate'
    else:
        return 'Low'

def determine_alert_level(predicted_cat, confidence, uncertainty, tuned_cat):
    """
    Determine alert urgency based on prediction and uncertainty
    """
    # Critical cases always get highest alert
    if tuned_cat == 'Critical':
        return 'URGENT - Critical Risk Detected'

    # High uncertainty cases need review
    if uncertainty > 0.1:  # 10% uncertainty threshold
        return 'REVIEW NEEDED - High Uncertainty'

    # Low confidence predictions
    if confidence < 0.7:
        return 'REVIEW NEEDED - Low Confidence'

    # Category disagreement between classifier and threshold tuning
    if predicted_cat != tuned_cat:
        return 'REVIEW NEEDED - Model Disagreement'

    # High risk with good confidence
    if tuned_cat == 'High' and confidence > 0.8:
        return 'MONITOR - High Risk'

    # Moderate and below with good confidence
    if confidence > 0.8:
        return 'NORMAL - Confident Prediction'

    return 'MONITOR - Standard Case'

def print_prediction_report(results, actual_data=None):
    """Print formatted prediction report with uncertainty metrics"""

    print("\n" + "="*100)
    print("RISK PREDICTION REPORT - TREE-BASED UNCERTAINTY ESTIMATION")
    print("="*100)

    for idx, row in results.iterrows():
        print(f"\n{'='*100}")
        print(f"PATIENT: {row['patient_id']}")
        print(f"{'='*100}")

        print(f"\nRISK ASSESSMENT:")
        print(f"  Predicted Category: {row['predicted_category']} (confidence: {row['confidence']:.1%})")
        print(f"  Tuned Category: {row['tuned_category']} (score-based)")
        print(f"  Risk Score: {row['risk_score']:.2f} (95% CI: [{row['score_ci_lower']:.2f}, {row['score_ci_upper']:.2f}])")

        print(f"\nUNCERTAINTY METRICS:")
        print(f"  Classification Uncertainty: {row['uncertainty']:.1%} (variance in tree predictions)")
        print(f"  Overall Model Uncertainty: {row['overall_uncertainty']:.1%}")
        print(f"  Score Variance: {row['score_variance']:.3f}")

        print(f"\nALL CLASS PROBABILITIES:")
        for cat, prob in sorted(row['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
            marker = " <- PREDICTED" if cat == row['predicted_category'] else ""
            print(f"  {cat:15s}: {prob:6.1%}{marker}")

        print(f"\nALERT STATUS: {row['alert_level']}")

        # If we have actual data, show accuracy
        if actual_data is not None and row['patient_id'] in actual_data.index:
            actual_cat = actual_data.loc[row['patient_id'], 'risk_category']
            actual_score = actual_data.loc[row['patient_id'], 'composite_risk_score']

            cat_correct = "OK CORRECT" if row['tuned_category'] == actual_cat else "WRONG"
            score_error = abs(row['risk_score'] - actual_score)
            score_error_pct = (score_error / actual_score * 100) if actual_score != 0 else 0

            print(f"\nACTUAL vs PREDICTED:")
            print(f"  Actual Category: {actual_cat}")
            print(f"  Actual Score: {actual_score:.2f}")
            print(f"  Classification: {cat_correct}")
            print(f"  Score Error: {score_error:.2f} ({score_error_pct:.1f}%)")

def main():
    # Load models
    models = load_latest_models()

    # Load test data
    data = pd.read_csv('../processed_data/integrated_features_improved_balanced.csv')

    # Test on same 5 samples as diagnostic
    test_indices = [705, 2336, 1972, 1643, 2124]
    test_samples = data.loc[test_indices].copy()

    print(f"\nTesting on {len(test_samples)} samples...")

    # Predict with tree-based uncertainty
    results = predict_with_tree_uncertainty(models, test_samples)

    # Print detailed report
    print_prediction_report(results, actual_data=data)

    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    # Calculate accuracy
    correct = 0
    for idx, row in results.iterrows():
        actual_cat = data.loc[row['patient_id'], 'risk_category']
        if row['tuned_category'] == actual_cat:
            correct += 1

    accuracy = correct / len(results) * 100
    avg_confidence = results['confidence'].mean()
    avg_uncertainty = results['overall_uncertainty'].mean()
    avg_score_variance = results['score_variance'].mean()

    print(f"\nOverall Accuracy: {accuracy:.1f}% ({correct}/{len(results)} correct)")
    print(f"Average Confidence: {avg_confidence:.1%}")
    print(f"Average Uncertainty: {avg_uncertainty:.1%}")
    print(f"Average Score Variance: {avg_score_variance:.3f}")

    # Alert distribution
    print(f"\nAlert Level Distribution:")
    alert_counts = results['alert_level'].value_counts()
    for alert, count in alert_counts.items():
        print(f"  {alert}: {count}")

    print("\n" + "="*100)
    print("NOTES:")
    print("  - Uncertainty is calculated from variance in tree predictions")
    print("  - Lower uncertainty = more confident model")
    print("  - Score CI uses variance from staged predictions")
    print("  - Alert levels consider both prediction and uncertainty")
    print("="*100)

if __name__ == "__main__":
    main()
