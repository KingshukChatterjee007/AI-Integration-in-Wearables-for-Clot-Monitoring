"""
Enhanced Model Comparison for Blood Clot Risk Prediction
Tests multiple ML algorithms and selects the best performer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_squared_error, r2_score, classification_report, confusion_matrix)

# Ensemble Methods
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor)

# Support Vector Machine
from sklearn.svm import SVC, SVR

# Neural Network
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Handle imbalanced data
from imblearn.over_sampling import SMOTE

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available - install with: pip install lightgbm")

# CatBoost
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available - install with: pip install catboost")

import warnings
warnings.filterwarnings('ignore')

class EnhancedModelComparison:
    def __init__(self, data_path='processed_data/integrated_features_improved_balanced.csv'):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train_class = None
        self.y_test_class = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}

    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        print("Loading data from:", self.data_path)
        self.data = pd.read_csv(self.data_path)

        # Remove zero-variance features
        variance = self.data.var(numeric_only=True)
        zero_var_cols = variance[variance == 0].index.tolist()
        if zero_var_cols:
            print(f"Removing {len(zero_var_cols)} zero-variance features")
            self.data = self.data.drop(columns=zero_var_cols)

        # Remove highly correlated features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr().abs()
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr = [column for column in corr_matrix.columns
                    if any(corr_matrix[column][upper_triangle[:, list(corr_matrix.columns).index(column)]] > 0.95)]

        if high_corr:
            print(f"Removing {len(high_corr)} highly correlated features")
            self.data = self.data.drop(columns=high_corr)

        # Prepare features and targets
        feature_cols = [col for col in self.data.columns
                       if col not in ['composite_risk_score', 'risk_category', 'subject_id', 'timestamp', 'activity']]

        # Ensure only numeric features
        X = self.data[feature_cols].select_dtypes(include=[np.number])
        self.feature_names = list(X.columns)  # Store feature names for metadata
        y_regression = self.data['composite_risk_score']
        y_classification = self.data['risk_category']

        # Encode categorical labels to numeric for algorithms that require it
        y_classification_encoded = self.label_encoder.fit_transform(y_classification)

        print(f"\nFinal feature set: {len(feature_cols)} features")
        print(f"Total samples: {len(X)}")
        print(f"Risk categories: {self.label_encoder.classes_}")

        # Stratified split
        self.X_train, self.X_test, self.y_train_class, self.y_test_class, self.y_train_reg, self.y_test_reg = \
            train_test_split(X, y_classification_encoded, y_regression, test_size=0.3,
                           random_state=42, stratify=y_classification_encoded)

        # Fit scaler on training data and transform both sets
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print("Features scaled using StandardScaler")

    def apply_smote(self):
        """Apply SMOTE for balanced training"""
        print("\nApplying SMOTE for class balancing...")
        # Adjust k_neighbors based on smallest class size
        unique, counts = np.unique(self.y_train_class, return_counts=True)
        min_class_size = counts.min()
        k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train_class)

        # Get corresponding regression targets
        original_indices = []
        for i, (x_bal, y_bal) in enumerate(zip(X_train_balanced.values, y_train_balanced)):
            # Find matching original sample
            matches = np.where((self.X_train.values == x_bal).all(axis=1))[0]
            if len(matches) > 0:
                original_indices.append(matches[0])
            else:
                # For synthetic samples, use mean of class
                class_mean = self.y_train_reg[self.y_train_class == y_bal].mean()
                original_indices.append(None)

        y_train_reg_balanced = []
        for idx, y_class in zip(original_indices, y_train_balanced):
            if idx is not None:
                y_train_reg_balanced.append(self.y_train_reg.iloc[idx])
            else:
                y_train_reg_balanced.append(self.y_train_reg[self.y_train_class == y_class].mean())

        return X_train_balanced, y_train_balanced, np.array(y_train_reg_balanced)

    def train_all_classifiers(self, X_train, y_train):
        """Train all classification algorithms"""
        classifiers = {}

        print("\n" + "="*80)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*80)

        # 1. Random Forest
        print("\n1. Random Forest...")
        classifiers['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight='balanced'
        )
        classifiers['Random Forest'].fit(X_train, y_train)

        # 2. Gradient Boosting
        print("2. Gradient Boosting...")
        classifiers['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        classifiers['Gradient Boosting'].fit(X_train, y_train)

        # 3. AdaBoost
        print("3. AdaBoost...")
        classifiers['AdaBoost'] = AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42
        )
        classifiers['AdaBoost'].fit(X_train, y_train)

        # 4. SVM
        print("4. Support Vector Machine...")
        classifiers['SVM'] = SVC(
            kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
        )
        classifiers['SVM'].fit(X_train, y_train)

        # 5. Neural Network
        print("5. Neural Network (MLP)...")
        classifiers['Neural Network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=42
        )
        classifiers['Neural Network'].fit(X_train, y_train)

        # 6. XGBoost
        if XGBOOST_AVAILABLE:
            print("6. XGBoost...")
            classifiers['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, eval_metric='mlogloss'
            )
            classifiers['XGBoost'].fit(X_train, y_train)

        # 7. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("7. LightGBM...")
            classifiers['LightGBM'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1
            )
            classifiers['LightGBM'].fit(X_train, y_train)

        # 8. CatBoost
        if CATBOOST_AVAILABLE:
            print("8. CatBoost...")
            classifiers['CatBoost'] = CatBoostClassifier(
                iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0
            )
            classifiers['CatBoost'].fit(X_train, y_train)

        return classifiers

    def train_all_regressors(self, X_train, y_train):
        """Train all regression algorithms"""
        regressors = {}

        print("\n" + "="*80)
        print("TRAINING REGRESSION MODELS")
        print("="*80)

        # 1. Random Forest
        print("\n1. Random Forest...")
        regressors['Random Forest'] = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )
        regressors['Random Forest'].fit(X_train, y_train)

        # 2. Gradient Boosting
        print("2. Gradient Boosting...")
        regressors['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
        )
        regressors['Gradient Boosting'].fit(X_train, y_train)

        # 3. AdaBoost
        print("3. AdaBoost...")
        regressors['AdaBoost'] = AdaBoostRegressor(
            n_estimators=200, learning_rate=0.5, random_state=42
        )
        regressors['AdaBoost'].fit(X_train, y_train)

        # 4. SVM
        print("4. Support Vector Machine...")
        regressors['SVM'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        regressors['SVM'].fit(X_train, y_train)

        # 5. Neural Network
        print("5. Neural Network (MLP)...")
        regressors['Neural Network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, random_state=42
        )
        regressors['Neural Network'].fit(X_train, y_train)

        # 6. XGBoost
        if XGBOOST_AVAILABLE:
            print("6. XGBoost...")
            regressors['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
            regressors['XGBoost'].fit(X_train, y_train)

        # 7. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("7. LightGBM...")
            regressors['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1
            )
            regressors['LightGBM'].fit(X_train, y_train)

        # 8. CatBoost
        if CATBOOST_AVAILABLE:
            print("8. CatBoost...")
            regressors['CatBoost'] = CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0
            )
            regressors['CatBoost'].fit(X_train, y_train)

        return regressors

    def evaluate_models(self, classifiers, regressors):
        """Evaluate all models and store results"""
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)

        # Evaluate classifiers
        for name, clf in classifiers.items():
            y_pred = clf.predict(self.X_test)

            self.results[name] = {
                'type': 'classification',
                'accuracy': accuracy_score(self.y_test_class, y_pred),
                'f1_score': f1_score(self.y_test_class, y_pred, average='weighted'),
                'precision': precision_score(self.y_test_class, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test_class, y_pred, average='weighted'),
                'predictions': y_pred,
                'model': clf
            }

        # Evaluate regressors
        for name, reg in regressors.items():
            y_pred = reg.predict(self.X_test)

            if name not in self.results:
                self.results[name] = {}

            self.results[name].update({
                'regression': True,
                'rmse': np.sqrt(mean_squared_error(self.y_test_reg, y_pred)),
                'r2': r2_score(self.y_test_reg, y_pred),
                'reg_predictions': y_pred,
                'reg_model': reg
            })

        self.print_results()

    def print_results(self):
        """Print formatted results"""
        print("\n" + "="*80)
        print("CLASSIFICATION RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*80)

        for name, metrics in sorted(self.results.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True):
            if 'f1_score' in metrics:
                print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
                      f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

        print("\n" + "="*80)
        print("REGRESSION RESULTS")
        print("="*80)
        print(f"{'Model':<20} {'RMSE':<12} {'R² Score':<12}")
        print("-"*80)

        for name, metrics in sorted(self.results.items(), key=lambda x: x[1].get('r2', -999), reverse=True):
            if 'regression' in metrics:
                print(f"{name:<20} {metrics['rmse']:<12.4f} {metrics['r2']:<12.4f}")

    def generate_visualizations(self):
        """Generate comprehensive comparison visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. Classification Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        models = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []

        for name, metrics in self.results.items():
            if 'f1_score' in metrics:
                models.append(name)
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1_score'])
                precisions.append(metrics['precision'])
                recalls.append(metrics['recall'])

        # Accuracy
        axes[0, 0].barh(models, accuracies, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Accuracy Score')
        axes[0, 0].set_title('Classification Accuracy Comparison')
        axes[0, 0].set_xlim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

        # F1-Score
        axes[0, 1].barh(models, f1_scores, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('F1-Score')
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_xlim(0, 1)
        for i, v in enumerate(f1_scores):
            axes[0, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

        # Precision
        axes[1, 0].barh(models, precisions, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Precision Score')
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].set_xlim(0, 1)
        for i, v in enumerate(precisions):
            axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')

        # Recall
        axes[1, 1].barh(models, recalls, color='pink', edgecolor='black')
        axes[1, 1].set_xlabel('Recall Score')
        axes[1, 1].set_title('Recall Comparison')
        axes[1, 1].set_xlim(0, 1)
        for i, v in enumerate(recalls):
            axes[1, 1].text(v + 0.01, i, f'{v:.3f}', va='center')

        plt.tight_layout()
        plt.savefig('integrated-images/01_classification_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/01_classification_metrics_comparison.png")
        plt.close()

        # 2. Regression Metrics Comparison (exclude extreme outliers)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        reg_models = []
        rmse_scores = []
        r2_scores = []

        for name, metrics in self.results.items():
            if 'regression' in metrics:
                # Exclude extreme outliers (Neural Network has RMSE=59, R²=-2089)
                if metrics['rmse'] < 10 and metrics['r2'] > -10:
                    reg_models.append(name)
                    rmse_scores.append(metrics['rmse'])
                    r2_scores.append(metrics['r2'])

        # RMSE (excluding outliers)
        ax1.barh(reg_models, rmse_scores, color='coral', edgecolor='black')
        ax1.set_xlabel('RMSE (Lower is Better)')
        ax1.set_title('Regression RMSE Comparison\n(Extreme outliers excluded)')
        for i, v in enumerate(rmse_scores):
            ax1.text(v + 0.05, i, f'{v:.3f}', va='center', fontsize=9)

        # R² Score (excluding outliers)
        colors = ['red' if r2 < 0 else 'lightblue' for r2 in r2_scores]
        ax2.barh(reg_models, r2_scores, color=colors, edgecolor='black')
        ax2.set_xlabel('R² Score (Higher is Better)')
        ax2.set_title('Regression R² Comparison\n(Extreme outliers excluded)')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        for i, v in enumerate(r2_scores):
            ax2.text(v + 0.01 if v >= 0 else v - 0.05, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('integrated-images/02_regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/02_regression_metrics_comparison.png")
        plt.close()

        # 3. Best Model Performance Summary
        best_clf = max(self.results.items(), key=lambda x: x[1].get('f1_score', 0))
        best_reg = max(self.results.items(), key=lambda x: x[1].get('r2', -999))

        fig = plt.figure(figsize=(16, 6))

        # Best classifier confusion matrix
        ax1 = plt.subplot(1, 2, 1)
        cm = confusion_matrix(self.y_test_class, best_clf[1]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Best Classifier: {best_clf[0]}\nF1-Score: {best_clf[1]["f1_score"]:.4f}')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        # Best regressor scatter plot
        ax2 = plt.subplot(1, 2, 2)
        ax2.scatter(self.y_test_reg, best_reg[1]['reg_predictions'], alpha=0.5)
        ax2.plot([self.y_test_reg.min(), self.y_test_reg.max()],
                [self.y_test_reg.min(), self.y_test_reg.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Risk Score')
        ax2.set_ylabel('Predicted Risk Score')
        ax2.set_title(f'Best Regressor: {best_reg[0]}\nR²: {best_reg[1]["r2"]:.4f}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/03_best_models_performance.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/03_best_models_performance.png")
        plt.close()

        # 4. Overall Model Ranking (with memory error handling)
        try:
            import gc
            plt.close('all')
            gc.collect()

            # Use minimal figure settings
            fig = plt.figure(figsize=(8, 5), dpi=80)
            ax = fig.add_subplot(111)

            # Calculate composite score
            composite_scores = {}
            f1_max = max([m.get('f1_score', 0) for m in self.results.values()])
            r2_max = max([m.get('r2', 0) for m in self.results.values()])

            for name, metrics in self.results.items():
                if 'f1_score' in metrics and 'r2' in metrics:
                    f1_norm = metrics['f1_score'] / f1_max if f1_max > 0 else 0
                    r2_norm = metrics['r2'] / r2_max if r2_max > 0 else 0
                    composite_scores[name] = (f1_norm * 0.5 + r2_norm * 0.5)

            sorted_models = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            model_names = [m[0] for m in sorted_models]
            scores = [m[1] for m in sorted_models]

            ax.barh(model_names, scores, color='mediumpurple', edgecolor='black')
            ax.set_xlabel('Composite Score')
            ax.set_title('Overall Model Ranking', fontsize=12)
            ax.set_xlim(0, 1)

            for i, score in enumerate(scores):
                ax.text(score + 0.02, i, f'{score:.2f}', va='center', fontsize=8)

            plt.tight_layout(pad=0.5)
            # Save with minimal settings
            fig.savefig('integrated-images/04_overall_model_ranking.png',
                       dpi=80, bbox_inches='tight', format='png')
            print("Saved: integrated-images/04_overall_model_ranking.png")
            plt.close(fig)
            gc.collect()
        except (MemoryError, Exception) as e:
            print(f"⚠ Skipped ranking plot due to: {type(e).__name__}")
            plt.close('all')

        print("\nAll critical visualizations generated successfully!")

        # Generate additional analysis plots (05-08)
        self.generate_analysis_plots(best_clf, best_reg)

        return best_clf[0], best_reg[0]

    def generate_analysis_plots(self, best_clf, best_reg):
        """Generate additional analysis visualizations (images 05-08)"""
        print("\n" + "="*80)
        print("GENERATING ADDITIONAL ANALYSIS PLOTS")
        print("="*80)

        # 5. Feature Group Importance
        print("\nGenerating feature group importance...")
        plt.figure(figsize=(12, 6))

        rf_model = best_clf[1]['model']
        importances = rf_model.feature_importances_

        # Group features by type
        feature_groups = {
            'demographic_features': [],
            'risk_features': [],
            'pleth_features': [],
            'temp_features': [],
            'ecg_features': [],
            'motion_features': [],
            'vital_changes': []
        }

        feature_names = self.X_train.columns
        for idx, fname in enumerate(feature_names):
            fname_lower = fname.lower()
            if any(x in fname_lower for x in ['age', 'bmi', 'weight', 'gender']):
                feature_groups['demographic_features'].append(importances[idx])
            elif 'risk' in fname_lower:
                feature_groups['risk_features'].append(importances[idx])
            elif 'pleth' in fname_lower:
                feature_groups['pleth_features'].append(importances[idx])
            elif 'temp' in fname_lower:
                feature_groups['temp_features'].append(importances[idx])
            elif 'ecg' in fname_lower:
                feature_groups['ecg_features'].append(importances[idx])
            elif 'accel' in fname_lower or 'motion' in fname_lower:
                feature_groups['motion_features'].append(importances[idx])
            elif 'change' in fname_lower:
                feature_groups['vital_changes'].append(importances[idx])

        group_importances = {k: sum(v) if v else 0 for k, v in feature_groups.items()}
        sorted_groups = sorted(group_importances.items(), key=lambda x: x[1], reverse=True)

        colors = ['purple', 'brown', 'orange', 'green', 'blue', 'red', 'pink']
        plt.barh([g[0].replace('_', ' ').title() for g in sorted_groups],
                [g[1] for g in sorted_groups],
                color=colors[:len(sorted_groups)],
                edgecolor='black')

        for i, (group, imp) in enumerate(sorted_groups):
            plt.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontweight='bold')

        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Feature Group Importance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('integrated-images/05_feature_group_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/05_feature_group_importance.png")
        plt.close()

        # 6. Top Individual Features
        print("Generating top features plot...")
        plt.figure(figsize=(12, 8))

        top_indices = np.argsort(importances)[-15:]
        top_importances = importances[top_indices]
        top_features = [feature_names[i] for i in top_indices]

        plt.barh(range(len(top_indices)), top_importances, color='coral', edgecolor='black')
        plt.yticks(range(len(top_indices)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 15 Most Important Individual Features', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        for i, imp in enumerate(top_importances):
            plt.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('integrated-images/06_top_features.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/06_top_features.png")
        plt.close()

        # 7. Risk by Activity and Subject
        print("Generating risk by activity/subject...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Activity risk
        if 'activity' in self.data.columns:
            activity_risk = self.data.groupby('activity')['composite_risk_score'].mean().sort_values(ascending=False)
            ax1.bar(activity_risk.index, activity_risk.values, color='lightgreen', edgecolor='black')
            ax1.set_xlabel('Activity Type', fontsize=11)
            ax1.set_ylabel('Average Risk Score', fontsize=11)
            ax1.set_title('Average Risk Score by Activity', fontsize=12, fontweight='bold')

            for i, (act, risk) in enumerate(activity_risk.items()):
                ax1.text(i, risk + 0.05, f'{risk:.3f}', ha='center', fontweight='bold')

        # Top risk subjects
        if 'subject_id' in self.data.columns:
            subject_risk = self.data.groupby('subject_id')['composite_risk_score'].mean().sort_values(ascending=False).head(10)
            ax2.bar(range(len(subject_risk)), subject_risk.values, color='lightcoral', edgecolor='black')
            ax2.set_xticks(range(len(subject_risk)))
            ax2.set_xticklabels([f"{s}\n({subject_risk[s]:.2f})" for s in subject_risk.index], fontsize=9)
            ax2.set_xlabel('Subject Rank (Top 10 Risk)', fontsize=11)
            ax2.set_ylabel('Average Risk Score', fontsize=11)
            ax2.set_title('Top 10 Highest Risk Subjects', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('integrated-images/07_risk_by_activity_subject.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/07_risk_by_activity_subject.png")
        plt.close()

        # 8. Clinical Insights
        print("Generating clinical insights...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Age vs Risk
        if 'age' in self.data.columns:
            ax1.scatter(self.data['age'], self.data['composite_risk_score'], alpha=0.5, c='purple')
            ax1.set_xlabel('Age (years)', fontsize=11)
            ax1.set_ylabel('Risk Score', fontsize=11)
            ax1.set_title('Risk Score vs Age Distribution', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # Prediction Residuals
        predictions = best_reg[1]['reg_predictions']
        residuals = self.y_test_reg - predictions
        ax2.scatter(predictions, residuals, alpha=0.5, c='red')
        ax2.axhline(y=0, color='blue', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values', fontsize=11)
        ax2.set_ylabel('Residuals', fontsize=11)
        ax2.set_title('Prediction Residuals Analysis', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('integrated-images/08_clinical_insights.png', dpi=300, bbox_inches='tight')
        print("Saved: integrated-images/08_clinical_insights.png")
        plt.close()

        print("\nAll analysis plots (05-08) generated successfully!")

    def save_best_models(self, best_clf_name, best_reg_name):
        """Save the best trained models to disk for future use"""

        # Create models directory if it doesn't exist
        models_dir = 'trained_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"\nCreated directory: {models_dir}/")

        print("\n" + "="*80)
        print("SAVING TRAINED MODELS")
        print("="*80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best classifier
        if best_clf_name in self.results and 'model' in self.results[best_clf_name]:
            clf_path = os.path.join(models_dir, f'best_classifier_{best_clf_name}_{timestamp}.pkl')
            joblib.dump(self.results[best_clf_name]['model'], clf_path)
            print(f"\nSaved Best Classifier ({best_clf_name}):")
            print(f"  -> {clf_path}")
            print(f"  Accuracy: {self.results[best_clf_name]['accuracy']:.4f}")
            print(f"  F1-Score: {self.results[best_clf_name]['f1_score']:.4f}")

        # Save best regressor
        if best_reg_name in self.results and 'reg_model' in self.results[best_reg_name]:
            reg_path = os.path.join(models_dir, f'best_regressor_{best_reg_name}_{timestamp}.pkl')
            joblib.dump(self.results[best_reg_name]['reg_model'], reg_path)
            print(f"\nSaved Best Regressor ({best_reg_name}):")
            print(f"  -> {reg_path}")
            print(f"  R² Score: {self.results[best_reg_name]['r2']:.4f}")
            print(f"  RMSE: {self.results[best_reg_name]['rmse']:.4f}")

        # Save preprocessing objects
        scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"\nSaved StandardScaler:")
        print(f"  -> {scaler_path}")

        encoder_path = os.path.join(models_dir, f'label_encoder_{timestamp}.pkl')
        joblib.dump(self.label_encoder, encoder_path)
        print(f"\nSaved LabelEncoder:")
        print(f"  -> {encoder_path}")

        # Save model metadata
        metadata = {
            'best_classifier': best_clf_name,
            'best_regressor': best_reg_name,
            'classifier_metrics': {
                'accuracy': self.results[best_clf_name]['accuracy'],
                'f1_score': self.results[best_clf_name]['f1_score'],
                'precision': self.results[best_clf_name]['precision'],
                'recall': self.results[best_clf_name]['recall']
            },
            'regressor_metrics': {
                'r2': self.results[best_reg_name]['r2'],
                'rmse': self.results[best_reg_name]['rmse']
            },
            'feature_columns': self.feature_names,
            'risk_categories': list(self.label_encoder.classes_),
            'timestamp': timestamp
        }

        metadata_path = os.path.join(models_dir, f'model_metadata_{timestamp}.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"\nSaved Model Metadata:")
        print(f"  -> {metadata_path}")
        print(f"  (Includes feature names, risk categories, and performance metrics)")

        print("\n" + "="*80)
        print("Models saved successfully! Use these files for deployment.")
        print("="*80)

        return models_dir

    def run_complete_analysis(self):
        """Run complete enhanced model comparison"""
        print("="*80)
        print("ENHANCED ML MODEL COMPARISON FOR BLOOD CLOT PREDICTION")
        print("="*80)

        # Load data
        self.load_and_prepare_data()

        # Apply SMOTE
        X_train_balanced, y_train_class_balanced, y_train_reg_balanced = self.apply_smote()

        # Train models
        classifiers = self.train_all_classifiers(X_train_balanced, y_train_class_balanced)
        regressors = self.train_all_regressors(X_train_balanced, y_train_reg_balanced)

        # Evaluate
        self.evaluate_models(classifiers, regressors)

        # Visualize
        best_clf, best_reg = self.generate_visualizations()

        # Final recommendation
        print("\n" + "="*80)
        print("FINAL RECOMMENDATIONS")
        print("="*80)
        print(f"\nBEST CLASSIFIER: {best_clf}")
        print(f"   F1-Score: {self.results[best_clf]['f1_score']:.4f}")
        print(f"   Accuracy: {self.results[best_clf]['accuracy']:.4f}")

        print(f"\nBEST REGRESSOR: {best_reg}")
        print(f"   R² Score: {self.results[best_reg]['r2']:.4f}")
        print(f"   RMSE: {self.results[best_reg]['rmse']:.4f}")

        # Save best models to disk
        models_dir = self.save_best_models(best_clf, best_reg)

        print("\n" + "="*80)
        print("Analysis complete! Check 'integrated-images/' for visualizations")
        print(f"Trained models saved in '{models_dir}/' directory")
        print("="*80)

        return self.results, best_clf, best_reg

if __name__ == "__main__":
    comparison = EnhancedModelComparison()
    results, best_classifier, best_regressor = comparison.run_complete_analysis()
