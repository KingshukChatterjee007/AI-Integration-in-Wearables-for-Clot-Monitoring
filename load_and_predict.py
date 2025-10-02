"""
Demonstration: Load Trained Models and Make Predictions
Shows how to use the saved models for real-world deployment
"""

import joblib
import pandas as pd
import numpy as np
import os
import glob

def load_latest_models(models_dir='trained_models'):
    """Load the most recently saved models"""

    print("="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    # Find latest timestamp
    metadata_files = glob.glob(os.path.join(models_dir, 'model_metadata_*.pkl'))
    if not metadata_files:
        raise FileNotFoundError(f"No saved models found in '{models_dir}/' directory")

    # Get most recent metadata file
    latest_metadata_file = max(metadata_files, key=os.path.getctime)
    metadata = joblib.load(latest_metadata_file)
    timestamp = metadata['timestamp']

    print(f"\nOK Found models from: {timestamp}")

    # Load classifier
    clf_name = metadata['best_classifier']
    clf_file = os.path.join(models_dir, f'best_classifier_{clf_name}_{timestamp}.pkl')
    classifier = joblib.load(clf_file)
    print(f"\nOK Loaded Classifier: {clf_name}")
    print(f"  Accuracy: {metadata['classifier_metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {metadata['classifier_metrics']['f1_score']:.4f}")

    # Load regressor
    reg_name = metadata['best_regressor']
    reg_file = os.path.join(models_dir, f'best_regressor_{reg_name}_{timestamp}.pkl')
    regressor = joblib.load(reg_file)
    print(f"\nOK Loaded Regressor: {reg_name}")
    print(f"  R² Score: {metadata['regressor_metrics']['r2']:.4f}")
    print(f"  RMSE: {metadata['regressor_metrics']['rmse']:.4f}")

    # Load preprocessing objects
    scaler_file = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
    scaler = joblib.load(scaler_file)
    print(f"\nOK Loaded StandardScaler")

    encoder_file = os.path.join(models_dir, f'label_encoder_{timestamp}.pkl')
    label_encoder = joblib.load(encoder_file)
    print(f"\nOK Loaded LabelEncoder")
    print(f"  Risk Categories: {list(label_encoder.classes_)}")

    print("\n" + "="*80)

    return {
        'classifier': classifier,
        'regressor': regressor,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'metadata': metadata
    }

def predict_risk(models, patient_data):
    """
    Make predictions for new patient data

    Args:
        models: Dictionary containing loaded models
        patient_data: DataFrame with same features as training data

    Returns:
        Dictionary with classification and regression predictions
    """

    # Ensure features are in correct order
    feature_columns = models['metadata']['feature_columns']

    # Check if all required features are present
    missing_features = set(feature_columns) - set(patient_data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Select and order features correctly
    X = patient_data[feature_columns]

    # Scale features (same as training) and keep as DataFrame
    X_scaled_array = models['scaler'].transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=feature_columns, index=X.index)

    # Classification prediction (risk category)
    class_pred_encoded = models['classifier'].predict(X_scaled)
    class_pred = models['label_encoder'].inverse_transform(class_pred_encoded)

    # Get prediction probabilities
    class_proba = models['classifier'].predict_proba(X_scaled)

    # Regression prediction (continuous risk score)
    risk_score = models['regressor'].predict(X_scaled)

    return {
        'risk_category': class_pred,
        'risk_score': risk_score,
        'category_probabilities': class_proba,
        'risk_categories': models['label_encoder'].classes_
    }

def demo_prediction():
    """Demonstrate prediction on sample data"""

    # Load models
    models = load_latest_models()

    print("\n" + "="*80)
    print("DEMONSTRATION: PREDICTING ON TEST DATA")
    print("="*80)

    # Load some sample data from the dataset
    data = pd.read_csv('processed_data/integrated_features_improved_balanced.csv')

    # Take 100 random samples for comprehensive testing
    sample_data = data.sample(n=100, random_state=42)

    # Extract features (remove target columns)
    feature_cols = [col for col in sample_data.columns
                   if col not in ['composite_risk_score', 'risk_category', 'subject_id', 'timestamp', 'activity']]

    X_sample = sample_data[feature_cols].select_dtypes(include=[np.number])

    # Make predictions
    predictions = predict_risk(models, X_sample)

    # Calculate overall statistics
    correct = 0
    total = len(X_sample)

    # Per-category statistics
    category_stats = {}
    for cat in predictions['risk_categories']:
        category_stats[cat] = {'correct': 0, 'total': 0, 'predicted': 0}

    # Display results summary
    print("\n" + "-"*80)
    print("PREDICTION RESULTS SUMMARY (100 Samples)")
    print("-"*80)

    for i in range(len(X_sample)):
        actual_category = sample_data.iloc[i]['risk_category']
        predicted_category = predictions['risk_category'][i]

        # Update statistics
        category_stats[actual_category]['total'] += 1
        category_stats[predicted_category]['predicted'] += 1

        if predicted_category == actual_category:
            correct += 1
            category_stats[actual_category]['correct'] += 1

    # Print overall accuracy
    overall_accuracy = (correct / total) * 100
    print(f"\nOVERALL ACCURACY: {correct}/{total} = {overall_accuracy:.2f}%")
    print(f"OVERALL ERROR: {total - correct}/{total} = {100 - overall_accuracy:.2f}%")

    # Print per-category performance
    print("\n" + "-"*80)
    print("PER-CATEGORY PERFORMANCE")
    print("-"*80)
    print(f"{'Category':<15} {'Actual':<10} {'Predicted':<12} {'Correct':<10} {'Recall':<12}")
    print("-"*80)

    for cat in ['Critical', 'High', 'Moderate', 'Low-Moderate', 'Low']:
        if cat in category_stats:
            stats = category_stats[cat]
            actual_count = stats['total']
            predicted_count = stats['predicted']
            correct_count = stats['correct']
            recall = (correct_count / actual_count * 100) if actual_count > 0 else 0
            print(f"{cat:<15} {actual_count:<10} {predicted_count:<12} {correct_count:<10} {recall:.2f}%")

    # Show confusion examples (first 10)
    print("\n" + "-"*80)
    print("SAMPLE PREDICTIONS (First 10)")
    print("-"*80)

    for i in range(min(10, len(X_sample))):
        actual_category = sample_data.iloc[i]['risk_category']
        predicted_category = predictions['risk_category'][i]
        actual_score = sample_data.iloc[i]['composite_risk_score']
        predicted_score = predictions['risk_score'][i]

        match = "CORRECT" if predicted_category == actual_category else "WRONG"

        print(f"\nSample {i+1}: [{match}]")
        print(f"  Predicted: {predicted_category} (score: {predicted_score:.2f})")
        print(f"  Actual:    {actual_category} (score: {actual_score:.2f})")

        # Show top 2 probabilities
        probs = predictions['category_probabilities'][i]
        sorted_idx = np.argsort(probs)[::-1][:2]
        print(f"  Top Probs: {predictions['risk_categories'][sorted_idx[0]]}: {probs[sorted_idx[0]]:.1%}, " +
              f"{predictions['risk_categories'][sorted_idx[1]]}: {probs[sorted_idx[1]]:.1%}")

    print("\n" + "="*80)
    print("Comprehensive testing complete!")
    print("="*80)

if __name__ == "__main__":
    demo_prediction()
