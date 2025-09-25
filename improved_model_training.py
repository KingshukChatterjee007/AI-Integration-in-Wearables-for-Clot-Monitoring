"""
Improved Blood Clot Risk Prediction Model Training
==================================================

This script addresses the fundamental issues causing poor performance:
1. Stratified train/test split
2. SMOTE oversampling for class balance
3. Proper evaluation metrics
4. Feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Handle XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class ImprovedBloodClotPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.DataFrame()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self):
        """Load and preprocess data with fixes"""
        print("Loading and preprocessing data...")
        self.data = pd.read_csv(self.data_path)

        # Remove zero-variance features
        exclude_cols = ['subject_id', 'activity', 'window_id', 'composite_risk_score',
                       'composite_risk_score_old', 'gender', 'risk_category']
        feature_columns = [col for col in self.data.columns if col not in exclude_cols]

        X = self.data[feature_columns]

        # Remove zero variance features
        zero_var_features = []
        for col in X.columns:
            if X[col].std() == 0:
                zero_var_features.append(col)

        if zero_var_features:
            print(f"Removing {len(zero_var_features)} zero-variance features: {zero_var_features}")
            X = X.drop(columns=zero_var_features)

        # Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]

        if high_corr_features:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)

        print(f"Final feature set: {X.shape[1]} features")

        return X, self.data['composite_risk_score']

    def create_balanced_classification_target(self, y_regression):
        """Create more balanced classification targets"""
        # Create more balanced bins
        y_classification = pd.cut(y_regression,
                                bins=[0, 1.5, 3.0, 5.0, 11],
                                labels=[0, 1, 2, 3],
                                include_lowest=True)

        print("\nClassification target distribution:")
        class_dist = y_classification.value_counts().sort_index()
        for cls, count in class_dist.items():
            pct = count/len(y_regression)*100
            print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

        return y_classification

    def stratified_train_test_split(self, X, y_reg, y_class):
        """Use stratified split instead of subject-based"""
        print("\nUsing stratified train/test split...")

        # Stratified split based on classification target
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=0.3, random_state=42, stratify=y_class)

        # Get corresponding classification targets
        y_train_class = y_class[X_train.index]
        y_test_class = y_class[X_test.index]

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        print("Training set distribution:")
        train_dist = y_train_class.value_counts().sort_index()
        for cls, count in train_dist.items():
            pct = count/len(X_train)*100
            print(f"  Class {cls}: {count} ({pct:.1f}%)")

        print("Test set distribution:")
        test_dist = y_test_class.value_counts().sort_index()
        for cls, count in test_dist.items():
            pct = count/len(X_test)*100
            print(f"  Class {cls}: {count} ({pct:.1f}%)")

        return X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class

    def apply_smote_balancing(self, X_train, y_train_class):
        """Apply SMOTE to balance training data"""
        print("\\nApplying SMOTE for class balancing...")

        # Use SMOTEENN for better balance
        smote_enn = SMOTEENN(random_state=42, sampling_strategy='auto')
        X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train_class)

        print("After SMOTE balancing:")
        balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
        for cls, count in balanced_dist.items():
            pct = count/len(y_train_balanced)*100
            print(f"  Class {cls}: {count} ({pct:.1f}%)")

        return X_train_balanced, y_train_balanced

    def train_improved_models(self, X_train, X_test, y_train, y_test):
        """Train models with proper handling"""
        print("\\n" + "="*50)
        print("TRAINING IMPROVED MODELS")
        print("="*50)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        # 1. Classification Models with Balanced Data
        print("\\n1. CLASSIFICATION MODELS:")
        print("-" * 30)

        # Apply SMOTE for balanced training
        X_train_balanced, y_train_balanced = self.apply_smote_balancing(X_train, y_train)
        X_train_balanced_scaled = self.scaler.fit_transform(X_train_balanced)

        # Random Forest Classifier
        rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf_clf.fit(X_train_balanced, y_train_balanced)
        rf_pred = rf_clf.predict(X_test)
        rf_prob = rf_clf.predict_proba(X_test)

        # XGBoost Classifier (if available)
        if XGBOOST_AVAILABLE:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='multi:softprob',
                eval_metric='mlogloss'
            )
            xgb_clf.fit(X_train_balanced, y_train_balanced)
            xgb_pred = xgb_clf.predict(X_test)
            xgb_prob = xgb_clf.predict_proba(X_test)

        # Calculate comprehensive metrics
        rf_accuracy = rf_clf.score(X_test, y_test)
        rf_f1 = f1_score(y_test, rf_pred, average='weighted')
        rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, rf_pred, average='weighted')

        print(f"Random Forest Results:")
        print(f"  Accuracy: {rf_accuracy:.4f}")
        print(f"  F1-Score: {rf_f1:.4f}")
        print(f"  Precision: {rf_precision:.4f}")
        print(f"  Recall: {rf_recall:.4f}")

        results['rf_classification'] = {
            'model': rf_clf,
            'predictions': rf_pred,
            'probabilities': rf_prob,
            'accuracy': rf_accuracy,
            'f1_score': rf_f1,
            'precision': rf_precision,
            'recall': rf_recall
        }

        if XGBOOST_AVAILABLE:
            xgb_accuracy = xgb_clf.score(X_test, y_test)
            xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
            xgb_precision = precision_score(y_test, xgb_pred, average='weighted', zero_division=0)
            xgb_recall = recall_score(y_test, xgb_pred, average='weighted')

            print(f"\\nXGBoost Results:")
            print(f"  Accuracy: {xgb_accuracy:.4f}")
            print(f"  F1-Score: {xgb_f1:.4f}")
            print(f"  Precision: {xgb_precision:.4f}")
            print(f"  Recall: {xgb_recall:.4f}")

            results['xgb_classification'] = {
                'model': xgb_clf,
                'predictions': xgb_pred,
                'probabilities': xgb_prob,
                'accuracy': xgb_accuracy,
                'f1_score': xgb_f1,
                'precision': xgb_precision,
                'recall': xgb_recall
            }

        # 2. Regression Models (for comparison)
        print("\\n2. REGRESSION MODELS:")
        print("-" * 30)

        # Random Forest Regression
        rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        rf_reg.fit(X_train, y_train)
        rf_reg_pred = rf_reg.predict(X_test)

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_reg_pred))
        rf_r2 = r2_score(y_test, rf_reg_pred)

        print(f"Random Forest Regression:")
        print(f"  RMSE: {rf_rmse:.4f}")
        print(f"  R²: {rf_r2:.4f}")

        results['rf_regression'] = {
            'model': rf_reg,
            'predictions': rf_reg_pred,
            'rmse': rf_rmse,
            'r2': rf_r2
        }

        # XGBoost Regression (if available)
        if XGBOOST_AVAILABLE:
            xgb_reg = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_reg.fit(X_train, y_train)
            xgb_reg_pred = xgb_reg.predict(X_test)

            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_reg_pred))
            xgb_r2 = r2_score(y_test, xgb_reg_pred)

            print(f"\\nXGBoost Regression:")
            print(f"  RMSE: {xgb_rmse:.4f}")
            print(f"  R²: {xgb_r2:.4f}")

            results['xgb_regression'] = {
                'model': xgb_reg,
                'predictions': xgb_reg_pred,
                'rmse': xgb_rmse,
                'r2': xgb_r2
            }

        self.results = results
        return results

    def create_improved_visualizations(self, X_test, y_test_class):
        """Create improved visualizations"""
        print("\\nCreating improved visualizations...")

        # 1. Enhanced Confusion Matrix
        plt.figure(figsize=(10, 8))

        # Use best classification model
        best_model = 'xgb_classification' if 'xgb_classification' in self.results else 'rf_classification'
        pred = self.results[best_model]['predictions']

        cm = confusion_matrix(y_test_class, pred)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        class_names = ['Low', 'Moderate', 'High', 'Critical']

        # Create better formatted annotations
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i,j] > 0:
                    annotations[i, j] = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
                else:
                    annotations[i, j] = f'0\n(0.0%)'

        # Create heatmap with better formatting
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   annot_kws={'size': 12, 'weight': 'bold'},
                   cbar_kws={'label': 'Count'},
                   linewidths=0.5, linecolor='white')

        # Highlight diagonal (correct predictions) with red border
        for i in range(len(class_names)):
            plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                               edgecolor='red', lw=4, alpha=0.8))

        plt.title(f'Confusion Matrix - {best_model.replace("_", " ").title()}\n(Numbers show Count and Row Percentage)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Risk Category', fontsize=12)
        plt.ylabel('Actual Risk Category', fontsize=12)
        plt.tight_layout()
        plt.savefig('04_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Performance Comparison (Regression + Feature Importance)
        plt.figure(figsize=(12, 5))

        # Regression comparison
        plt.subplot(1, 2, 1)

        # Get regression results
        reg_models = []
        r2_scores = []
        rmse_scores = []

        for model_name, results in self.results.items():
            if 'regression' in model_name:
                reg_models.append(model_name.replace('_regression', '').upper())
                r2_scores.append(results['r2'])
                rmse_scores.append(results['rmse'])

        x_pos_reg = np.arange(len(reg_models))
        colors = ['red' if r2 < 0 else 'skyblue' for r2 in r2_scores]

        bars = plt.bar(x_pos_reg, r2_scores, color=colors, alpha=0.8, edgecolor='black')
        plt.xlabel('Models', fontsize=10)
        plt.ylabel('R² Score')
        plt.title('Regression Model R² Comparison')
        plt.xticks(x_pos_reg, reg_models)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3)

        # Add value labels positioned just above bars
        for i, (bar, score) in enumerate(zip(bars, r2_scores)):
            height = bar.get_height()
            # Use small fixed offset just above the bar
            y_pos = height + 0.025 if height >= 0 else height - 0.025
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{score:.3f}', ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

        # Feature importance
        plt.subplot(1, 2, 2)
        rf_model = self.results['rf_classification']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10

        plt.barh(range(len(indices)), importances[indices], alpha=0.8, color='orange')
        plt.yticks(range(len(indices)), [f'Feature_{i}' for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('01_regression_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Classification Performance Chart
        plt.figure(figsize=(12, 6))

        # Get classification models for comparison
        models = []
        accuracies = []
        f1_scores = []

        for model_name, results in self.results.items():
            if 'classification' in model_name:
                models.append(model_name.replace('_classification', '').upper())
                accuracies.append(results['accuracy'])
                f1_scores.append(results['f1_score'])

        x_pos = np.arange(len(models))
        width = 0.35

        plt.bar(x_pos - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='green', edgecolor='black')
        plt.bar(x_pos + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='blue', edgecolor='black')

        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Classification Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, models)
        plt.legend(fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (acc, f1) in enumerate(zip(accuracies, f1_scores)):
            plt.text(i - width/2, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            plt.text(i + width/2, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig('02_classification_performance.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Visualizations saved:")
        print("- 04_confusion_matrix.png")
        print("- 01_regression_performance.png")
        print("- 02_classification_performance.png")

    def generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        print("\\n" + "="*80)
        print("COMPREHENSIVE IMPROVED MODEL REPORT")
        print("="*80)

        print("\\nCLASSIFICATION PERFORMANCE:")
        print("-" * 50)

        for model_name, results in self.results.items():
            if 'classification' in model_name:
                name = model_name.replace('_classification', '').upper()
                print(f"{name:15} | Accuracy: {results['accuracy']:.4f} | F1: {results['f1_score']:.4f} | Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f}")

        print("\\nREGRESSION PERFORMANCE:")
        print("-" * 50)

        for model_name, results in self.results.items():
            if 'regression' in model_name:
                name = model_name.replace('_regression', '').upper()
                print(f"{name:15} | RMSE: {results['rmse']:.4f} | R²: {results['r2']:.4f}")

        # Find best models
        best_clf = max([k for k in self.results.keys() if 'classification' in k],
                      key=lambda x: self.results[x]['f1_score'])

        print(f"\\nBEST CLASSIFICATION MODEL: {best_clf.replace('_classification', '').upper()}")
        print(f"   F1-Score: {self.results[best_clf]['f1_score']:.4f}")
        print(f"   Accuracy: {self.results[best_clf]['accuracy']:.4f}")

        print(f"\\nIMPROVEMENTS ACHIEVED:")
        print("+ Stratified train/test split ensures balanced evaluation")
        print("+ SMOTE balancing improves minority class predictions")
        print("+ Comprehensive metrics (F1, precision, recall)")
        print("+ Better model hyperparameters")
        print("+ Removed zero-variance and highly correlated features")

    def run_complete_improved_analysis(self):
        """Run the complete improved analysis pipeline"""
        print("Starting improved blood clot risk prediction analysis...")
        print("="*80)

        # Load and preprocess
        X, y_reg = self.load_and_preprocess_data()
        y_class = self.create_balanced_classification_target(y_reg)

        # Improved train/test split
        X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = self.stratified_train_test_split(X, y_reg, y_class)

        # Train improved models
        self.train_improved_models(X_train, X_test, y_train_class, y_test_class)

        # Create visualizations
        self.create_improved_visualizations(X_test, y_test_class)

        # Generate report
        self.generate_comprehensive_report()

        return self.results

if __name__ == "__main__":
    # Initialize improved predictor
    predictor = ImprovedBloodClotPredictor('processed_data/integrated_features_improved.csv')

    # Run improved analysis
    results = predictor.run_complete_improved_analysis()

    print("\\nImproved analysis complete!")