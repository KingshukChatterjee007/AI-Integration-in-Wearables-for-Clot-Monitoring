"""
Comprehensive Blood Clot Risk Prediction Model Training
Phase 1: Integrated Features Dataset Analysis

This script trains multiple ML models on the integrated_features.csv dataset
to predict composite_risk_score for blood clot monitoring in wearables.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Handle XGBoost import more gracefully
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class BloodClotRiskPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.data_path = data_path
        self.data = pd.DataFrame()  # Initialize as empty DataFrame instead of None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X = pd.DataFrame()  # Initialize feature DataFrame
        self.feature_groups = {}  # Initialize feature groups
        self.class_labels = {}

    def load_and_explore_data(self):
        """Load and perform initial exploration of the dataset"""
        print("Loading integrated_features.csv dataset...")
        self.data = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1] - 1}")
        print(f"Records: {self.data.shape[0]}")

        # Basic statistics
        print("\nTarget variable (composite_risk_score) statistics:")
        print(self.data['composite_risk_score'].describe())

        # Check distribution of target variable
        risk_value_counts = self.data['composite_risk_score'].value_counts().sort_index()
        print("\nRisk score value distribution:")
        for score, count in risk_value_counts.items():
            percentage = (count / len(self.data)) * 100
            print(f"Score {score}: {count} records ({percentage:.1f}%)")

        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        print(f"Total missing values: {missing_values}")

        # Subject distribution
        print(f"Number of unique subjects: {self.data['subject_id'].nunique()}")
        print(f"Activities: {self.data['activity'].unique()}")

        return self.data

    def prepare_features_targets(self):
        """Prepare features and targets for modeling"""

        # Define feature groups for analysis
        feature_groups = {
            'ecg_features': [col for col in self.data.columns if col.startswith('ecg_')],
            'pleth_features': [col for col in self.data.columns if col.startswith('pleth_')],
            'temp_features': [col for col in self.data.columns if col.startswith('temp_')],
            'motion_features': [col for col in self.data.columns if 'accel' in col or 'gyro' in col],
            'demographic_features': ['gender_encoded', 'height', 'weight', 'age', 'bmi'],
            'risk_features': [col for col in self.data.columns if 'risk' in col and col != 'composite_risk_score'],
            'vital_changes': [col for col in self.data.columns if 'change' in col]
        }

        # Prepare features (exclude non-feature columns)
        exclude_cols = ['subject_id', 'activity', 'window_id', 'composite_risk_score', 'gender']
        feature_columns = [col for col in self.data.columns if col not in exclude_cols]

        X = self.data[feature_columns]
        y_regression = self.data['composite_risk_score']

        # Create classification target (risk categories)
        y_classification = pd.cut(y_regression,
                                bins=[0, 1, 2, 3, 4],
                                labels=[0, 1, 2, 3],  # Use integer labels for XGBoost compatibility
                                include_lowest=True)

        # Store label mapping for interpretation
        self.class_labels = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Critical'}

        print(f"Features selected: {len(feature_columns)}")
        print("Risk category distribution:")
        risk_dist = y_classification.value_counts().sort_index()
        for code, count in risk_dist.items():
            label_name = self.class_labels[code] if code in self.class_labels else f"Class_{code}"
            print(f"{label_name} ({code}): {count}")

        return X, y_regression, y_classification, feature_groups

    def subject_based_split(self, X, y):
        """Split data by subjects to avoid data leakage"""

        # Get unique subjects
        subjects = self.data['subject_id'].unique()
        n_subjects = len(subjects)

        # Split subjects (not individual records)
        train_subjects = subjects[:int(0.7 * n_subjects)]  # 70% subjects for training
        test_subjects = subjects[int(0.7 * n_subjects):]   # 30% subjects for testing

        # Create train/test indices based on subjects
        train_idx = self.data['subject_id'].isin(train_subjects)
        test_idx = self.data['subject_id'].isin(test_subjects)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print(f"Training subjects: {len(train_subjects)}")
        print(f"Testing subjects: {len(test_subjects)}")
        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")

        return X_train, X_test, y_train, y_test, train_subjects, test_subjects

    def train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train regression models for continuous risk score prediction"""

        print("\nTraining regression models...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model configurations
        models_config = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }

        # Add XGBoost only if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        regression_results = {}

        for model_name, model in models_config.items():
            print(f"Training {model_name}...")

            # Use scaled features for Neural Network, original for tree-based models
            if model_name == 'Neural Network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

            # Handle edge case where r2 might be negative or very low
            if r2 < 0:
                r2_display = f"{r2:.4f} (negative)"
            else:
                r2_display = f"{r2:.4f}"

            regression_results[model_name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }

            print(f"{model_name} - RMSE: {rmse:.4f}, R2: {r2_display}")

        self.models['regression'] = models_config
        self.results['regression'] = regression_results

        return regression_results

    def train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train classification models for risk category prediction"""

        print("\nTraining classification models...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Model configurations
        models_config = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }

        # Add XGBoost only if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        classification_results = {}

        for model_name, model in models_config.items():
            print(f"Training {model_name} for classification...")

            # Use scaled features for Neural Network, original for tree-based models
            if model_name == 'Neural Network':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

            classification_results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': model.score(X_test if model_name != 'Neural Network' else X_test_scaled, y_test)
            }

            print(f"{model_name} - Accuracy: {classification_results[model_name]['accuracy']:.4f}")

        self.models['classification'] = models_config
        self.results['classification'] = classification_results

        return classification_results

    def analyze_feature_importance(self, X, feature_groups):
        """Analyze feature importance from trained models"""

        print("\nAnalyzing feature importance...")

        # Get feature importance from Random Forest regression
        rf_model = self.results['regression']['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Top 20 most important features
        top_features = feature_importance.head(20)

        print("Top 20 most important features:")
        for idx, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        # Group importance by feature type
        group_importance = {}
        for group_name, group_features in feature_groups.items():
            group_cols = [col for col in group_features if col in X.columns]
            if group_cols:
                group_imp = feature_importance[feature_importance['feature'].isin(group_cols)]['importance'].sum()
                group_importance[group_name] = group_imp

        print("\nFeature group importance:")
        for group, importance in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{group}: {importance:.4f}")

        return feature_importance, group_importance

    def create_visualizations(self, y_test_reg, y_test_class):
        """Create separate, clear visualizations for better readability"""

        print("\nCreating individual visualizations...")

        # 1. Model Performance Comparison (Regression)
        plt.figure(figsize=(10, 6))
        model_names = list(self.results['regression'].keys())
        r2_scores = [self.results['regression'][model]['r2'] for model in model_names]
        rmse_scores = [self.results['regression'][model]['rmse'] for model in model_names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        x_pos = np.arange(len(model_names))
        ax1.bar(x_pos, r2_scores, alpha=0.8, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Models', fontsize=12)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_title('Regression Model R² Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.grid(True, alpha=0.3)

        ax2.bar(x_pos, rmse_scores, alpha=0.8, color='orange', edgecolor='black')
        ax2.set_xlabel('Models', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('Regression Model RMSE Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/01_regression_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Classification Model Performance
        plt.figure(figsize=(10, 6))
        class_models = list(self.results['classification'].keys())
        accuracies = [self.results['classification'][model]['accuracy'] for model in class_models]

        x_pos_class = np.arange(len(class_models))
        plt.bar(x_pos_class, accuracies, alpha=0.8, color='green', edgecolor='black')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Classification Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos_class, class_models)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)

        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig('integrated-images/02_classification_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Actual vs Predicted and Risk Distribution
        best_reg_model = max(self.results['regression'].keys(),
                           key=lambda x: self.results['regression'][x]['r2'])
        best_predictions = self.results['regression'][best_reg_model]['predictions']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(y_test_reg, best_predictions, alpha=0.6, color='blue', edgecolor='black')
        ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Risk Score', fontsize=12)
        ax1.set_ylabel('Predicted Risk Score', fontsize=12)
        ax1.set_title(f'Actual vs Predicted Risk Score\n({best_reg_model})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.hist(self.data['composite_risk_score'], bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax2.set_xlabel('Composite Risk Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Risk Score Distribution in Dataset', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/03_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Confusion Matrix
        plt.figure(figsize=(8, 6))
        best_class_model = max(self.results['classification'].keys(),
                              key=lambda x: self.results['classification'][x]['accuracy'])
        best_class_pred = self.results['classification'][best_class_model]['predictions']

        cm = confusion_matrix(y_test_class, best_class_pred)
        unique_classes = sorted(set(y_test_class) | set(best_class_pred))
        class_names = [self.class_labels[cls] for cls in unique_classes]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_names, yticklabels=class_names,
                   annot_kws={'size': 14})
        plt.title(f'Confusion Matrix - {best_class_model}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)

        plt.tight_layout()
        plt.savefig('integrated-images/04_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Feature Group Importance
        plt.figure(figsize=(10, 6))
        _, group_importance = self.analyze_feature_importance(self.X, self.feature_groups)

        groups = list(group_importance.keys())
        importances = list(group_importance.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))

        plt.barh(groups, importances, color=colors, edgecolor='black')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature Groups', fontsize=12)
        plt.title('Feature Group Importance Analysis', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        for i, imp in enumerate(importances):
            plt.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('integrated-images/05_feature_group_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. Top Individual Features
        plt.figure(figsize=(12, 8))
        feature_importance, _ = self.analyze_feature_importance(self.X, self.feature_groups)
        top_15_features = feature_importance.head(15)

        plt.barh(range(len(top_15_features)), top_15_features['importance'].values,
                color='coral', edgecolor='black')
        plt.yticks(range(len(top_15_features)), [str(x) for x in top_15_features['feature'].values])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Top 15 Most Important Individual Features', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        for i, imp in enumerate(top_15_features['importance'].values):
            plt.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('integrated-images/06_top_features.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 7. Risk Analysis by Activity and Subject
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        activity_risk = self.data.groupby('activity')['composite_risk_score'].mean()
        ax1.bar(activity_risk.index, activity_risk.values, alpha=0.8, color='lightgreen', edgecolor='black')
        ax1.set_xlabel('Activity Type', fontsize=12)
        ax1.set_ylabel('Average Risk Score', fontsize=12)
        ax1.set_title('Average Risk Score by Activity', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        for i, val in enumerate(activity_risk.values):
            ax1.text(i, val + 0.005, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

        subject_risk = self.data.groupby('subject_id')['composite_risk_score'].mean().sort_values(ascending=False).head(10)
        ax2.bar(range(len(subject_risk)), subject_risk.values, alpha=0.8, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Subject Rank (Top 10 Risk)', fontsize=12)
        ax2.set_ylabel('Average Risk Score', fontsize=12)
        ax2.set_title('Top 10 Highest Risk Subjects', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(subject_risk)))
        ax2.set_xticklabels([f'{str(subj)}\n({val:.2f})' for subj, val in zip(subject_risk.index, subject_risk.values)],
                           rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/07_risk_by_activity_subject.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 8. Clinical Insights
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.scatter(self.data['age'], self.data['composite_risk_score'], alpha=0.6, color='purple', edgecolor='black')
        ax1.set_xlabel('Age (years)', fontsize=12)
        ax1.set_ylabel('Risk Score', fontsize=12)
        ax1.set_title('Risk Score vs Age Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        residuals = y_test_reg - best_predictions
        ax2.scatter(best_predictions, residuals, alpha=0.6, color='red', edgecolor='black')
        ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Prediction Residuals Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/08_clinical_insights.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\nAll visualizations created successfully in 'integrated-images/' folder:")
        print("integrated-images/01_regression_performance.png")
        print("integrated-images/02_classification_performance.png")
        print("integrated-images/03_prediction_vs_actual.png")
        print("integrated-images/04_confusion_matrix.png")
        print("integrated-images/05_feature_group_importance.png")
        print("integrated-images/06_top_features.png")
        print("integrated-images/07_risk_by_activity_subject.png")
        print("integrated-images/08_clinical_insights.png")

    def generate_model_summary(self):
        """Generate comprehensive model summary"""

        print("\n" + "="*80)
        print("COMPREHENSIVE BLOOD CLOT RISK PREDICTION MODEL SUMMARY")
        print("="*80)

        print(f"Dataset: {self.data.shape[0]} records, {self.data.shape[1]-1} features")
        print(f"Subjects: {self.data['subject_id'].nunique()}")
        print(f"Activities: {', '.join(self.data['activity'].unique())}")

        print("\nREGRESSION MODELS (Continuous Risk Score 0-4):")
        print("-" * 50)
        for model_name, results in self.results['regression'].items():
            print(f"{model_name:15} | RMSE: {results['rmse']:.4f} | R²: {results['r2']:.4f}")

        print("\nCLASSIFICATION MODELS (Risk Categories):")
        print("-" * 50)
        for model_name, results in self.results['classification'].items():
            print(f"{model_name:15} | Accuracy: {results['accuracy']:.4f}")

        # Best models - handle case where results might be empty
        if self.results.get('regression'):
            best_reg = max(self.results['regression'].keys(),
                          key=lambda x: self.results['regression'][x]['r2'])
            best_r2 = self.results['regression'][best_reg]['r2']
        else:
            best_reg = "No regression models"
            best_r2 = 0.0

        if self.results.get('classification'):
            best_class = max(self.results['classification'].keys(),
                            key=lambda x: self.results['classification'][x]['accuracy'])
            best_acc = self.results['classification'][best_class]['accuracy']
        else:
            best_class = "No classification models"
            best_acc = 0.0

        print(f"\nBEST REGRESSION MODEL: {best_reg} (R² = {best_r2:.4f})")
        print(f"BEST CLASSIFICATION MODEL: {best_class} (Accuracy = {best_acc:.4f})")

        print("\nCLINICAL INTERPRETATION:")
        print("-" * 50)
        print("• Risk Score 0-1: Low risk - routine monitoring")
        print("• Risk Score 1-2: Moderate risk - increased monitoring")
        print("• Risk Score 2-3: High risk - medical consultation recommended")
        print("• Risk Score 3-4: Critical risk - immediate medical attention")

        print("\nMODEL DEPLOYMENT READINESS:")
        print("-" * 50)

        if best_r2 > 0.8 and best_acc > 0.85:
            deployment_status = "READY FOR CLINICAL VALIDATION"
        elif best_r2 > 0.7 and best_acc > 0.8:
            deployment_status = "READY FOR PILOT TESTING"
        else:
            deployment_status = "NEEDS FURTHER IMPROVEMENT"

        print(f"Status: {deployment_status}")

    def run_complete_analysis(self):
        """Run the complete model training and analysis pipeline"""

        print("Starting comprehensive blood clot risk prediction analysis...")
        print("="*80)

        # Step 1: Load and explore data
        self.load_and_explore_data()

        # Step 2: Prepare features and targets
        self.X, self.y_reg, self.y_class, self.feature_groups = self.prepare_features_targets()

        # Step 3: Subject-based train-test split
        X_train, X_test, y_train_reg, y_test_reg, train_subj, test_subj = self.subject_based_split(self.X, self.y_reg)
        _, _, y_train_class, y_test_class, _, _ = self.subject_based_split(self.X, self.y_class)

        # Step 4: Train regression models
        self.train_regression_models(X_train, X_test, y_train_reg, y_test_reg)

        # Step 5: Train classification models
        self.train_classification_models(X_train, X_test, y_train_class, y_test_class)

        # Step 6: Feature importance analysis
        self.analyze_feature_importance(self.X, self.feature_groups)

        # Step 7: Create visualizations
        self.create_visualizations(y_test_reg, y_test_class)

        # Step 8: Generate summary
        self.generate_model_summary()

        return self.models, self.results


# Main execution
if __name__ == "__main__":

    # Initialize the predictor
    predictor = BloodClotRiskPredictor('processed_data/integrated_features.csv')

    # Run complete analysis
    models, results = predictor.run_complete_analysis()

    print("\nAnalysis complete! Check 'integrated-images/' folder for all visualizations.")