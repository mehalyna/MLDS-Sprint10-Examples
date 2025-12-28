"""
Task 7: Handling Imbalanced Data with Resampling Techniques
Demonstrates random oversampling, undersampling, SMOTE, and combination approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def create_imbalanced_dataset():
    """Create an imbalanced classification dataset"""
    print("Creating Imbalanced Dataset:")
    print("=" * 70)
    
    # Create imbalanced dataset (10:1 ratio)
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        flip_y=0.01,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Print class distribution
    class_counts = Counter(y)
    print(f"\nClass distribution:")
    print(f"  Class 0 (Majority): {class_counts[0]} samples ({class_counts[0]/len(y)*100:.1f}%)")
    print(f"  Class 1 (Minority): {class_counts[1]} samples ({class_counts[1]/len(y)*100:.1f}%)")
    print(f"  Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")
    
    return df, X, y


def visualize_class_distribution(y_original, y_resampled_dict, title):
    """Visualize class distributions before and after resampling"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    
    # Original distribution
    counts_original = Counter(y_original)
    axes[0].bar(['Class 0', 'Class 1'], 
                [counts_original[0], counts_original[1]], 
                color=['blue', 'red'], edgecolor='black', alpha=0.7)
    axes[0].set_title('Original Imbalanced')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].text(0, counts_original[0], f"{counts_original[0]}", ha='center', va='bottom')
    axes[0].text(1, counts_original[1], f"{counts_original[1]}", ha='center', va='bottom')
    
    # Resampled distributions
    for idx, (method, y_resampled) in enumerate(y_resampled_dict.items(), 1):
        if idx < len(axes):
            counts = Counter(y_resampled)
            axes[idx].bar(['Class 0', 'Class 1'], 
                         [counts[0], counts[1]], 
                         color=['blue', 'red'], edgecolor='black', alpha=0.7)
            axes[idx].set_title(method)
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].text(0, counts[0], f"{counts[0]}", ha='center', va='bottom')
            axes[idx].text(1, counts[1], f"{counts[1]}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'task7_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: task7_{title}.png")


def random_oversampling(X, y):
    """
    Apply random oversampling to balance the dataset
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\nRandom Oversampling:")
    print("=" * 70)
    print("Method: Randomly duplicate minority class samples until balanced")
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    print(f"\nOriginal distribution: {Counter(y)}")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"New dataset size: {len(y_resampled)} samples")
    
    return X_resampled, y_resampled


def random_undersampling(X, y):
    """
    Apply random undersampling to balance the dataset
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\nRandom Undersampling:")
    print("=" * 70)
    print("Method: Randomly remove majority class samples until balanced")
    
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    print(f"\nOriginal distribution: {Counter(y)}")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"New dataset size: {len(y_resampled)} samples")
    print(f"Samples removed: {len(y) - len(y_resampled)}")
    
    return X_resampled, y_resampled


def smote_oversampling(X, y):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\nSMOTE (Synthetic Minority Over-sampling Technique):")
    print("=" * 70)
    print("Method: Generate synthetic samples by interpolating between minority samples")
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"\nOriginal distribution: {Counter(y)}")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"New dataset size: {len(y_resampled)} samples")
    print(f"Synthetic samples created: {len(y_resampled) - len(y)}")
    
    return X_resampled, y_resampled


def smote_tomek(X, y):
    """
    Apply SMOTE + Tomek Links (combination approach)
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\nSMOTE + Tomek Links (Combination Approach):")
    print("=" * 70)
    print("Method: Apply SMOTE then remove Tomek links (borderline samples)")
    
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    
    print(f"\nOriginal distribution: {Counter(y)}")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"New dataset size: {len(y_resampled)} samples")
    
    return X_resampled, y_resampled


def smote_enn(X, y):
    """
    Apply SMOTE + Edited Nearest Neighbours
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\nSMOTE + ENN (Edited Nearest Neighbours):")
    print("=" * 70)
    print("Method: Apply SMOTE then clean using ENN (remove misclassified samples)")
    
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    
    print(f"\nOriginal distribution: {Counter(y)}")
    print(f"Resampled distribution: {Counter(y_resampled)}")
    print(f"New dataset size: {len(y_resampled)} samples")
    
    return X_resampled, y_resampled


def train_and_evaluate_model(X_train, X_test, y_train, y_test, method_name):
    """
    Train and evaluate a model on resampled data
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    method_name : str
        Name of the resampling method
        
    Returns:
    --------
    Dictionary with evaluation metrics
    """
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for minority class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for majority class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    results = {
        'Method': method_name,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }
    
    return results, model


def compare_resampling_methods(X, y):
    """
    Compare all resampling methods
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    """
    print("\n" + "=" * 70)
    print("Comparing Resampling Methods")
    print("=" * 70)
    
    # Split original data
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results_list = []
    resampled_data = {}
    
    # 1. No resampling (baseline)
    print("\n[1/6] Training on original imbalanced data (Baseline)...")
    results, _ = train_and_evaluate_model(X_train_orig, X_test, y_train_orig, y_test, 'No Resampling')
    results_list.append(results)
    
    # 2. Random Oversampling
    print("\n[2/6] Applying Random Oversampling...")
    X_train_ros, y_train_ros = random_oversampling(X_train_orig, y_train_orig)
    results, _ = train_and_evaluate_model(X_train_ros, X_test, y_train_ros, y_test, 'Random Oversampling')
    results_list.append(results)
    resampled_data['Random Oversampling'] = y_train_ros
    
    # 3. Random Undersampling
    print("\n[3/6] Applying Random Undersampling...")
    X_train_rus, y_train_rus = random_undersampling(X_train_orig, y_train_orig)
    results, _ = train_and_evaluate_model(X_train_rus, X_test, y_train_rus, y_test, 'Random Undersampling')
    results_list.append(results)
    resampled_data['Random Undersampling'] = y_train_rus
    
    # 4. SMOTE
    print("\n[4/6] Applying SMOTE...")
    X_train_smote, y_train_smote = smote_oversampling(X_train_orig, y_train_orig)
    results, _ = train_and_evaluate_model(X_train_smote, X_test, y_train_smote, y_test, 'SMOTE')
    results_list.append(results)
    resampled_data['SMOTE'] = y_train_smote
    
    # 5. SMOTE + Tomek
    print("\n[5/6] Applying SMOTE + Tomek Links...")
    X_train_smt, y_train_smt = smote_tomek(X_train_orig, y_train_orig)
    results, _ = train_and_evaluate_model(X_train_smt, X_test, y_train_smt, y_test, 'SMOTE + Tomek')
    results_list.append(results)
    resampled_data['SMOTE + Tomek'] = y_train_smt
    
    # 6. SMOTE + ENN
    print("\n[6/6] Applying SMOTE + ENN...")
    X_train_senn, y_train_senn = smote_enn(X_train_orig, y_train_orig)
    results, _ = train_and_evaluate_model(X_train_senn, X_test, y_train_senn, y_test, 'SMOTE + ENN')
    results_list.append(results)
    resampled_data['SMOTE + ENN'] = y_train_senn
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "=" * 70)
    print("Performance Comparison:")
    print("=" * 70)
    print("\n", results_df[['Method', 'Accuracy', 'F1-Score', 'ROC-AUC', 'Sensitivity', 'Specificity']].to_string(index=False))
    
    # Visualize class distributions
    print("\n" + "=" * 70)
    print("Visualizing Class Distributions:")
    print("=" * 70)
    visualize_class_distribution(y_train_orig, resampled_data, 'distributions')
    
    return results_df, resampled_data


def visualize_confusion_matrices(results_df):
    """Visualize confusion matrices for all methods"""
    print("\n" + "=" * 70)
    print("Creating Confusion Matrix Visualizations:")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, row in results_df.iterrows():
        cm = np.array([[row['TN'], row['FP']], 
                       [row['FN'], row['TP']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Pred 0', 'Pred 1'],
                   yticklabels=['True 0', 'True 1'])
        axes[idx].set_title(f"{row['Method']}\nF1: {row['F1-Score']:.3f}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('task7_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task7_confusion_matrices.png")


def visualize_metrics_comparison(results_df):
    """Visualize metrics comparison across methods"""
    print("\n" + "=" * 70)
    print("Creating Metrics Comparison Visualization:")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    methods = results_df['Method'].values
    x_pos = np.arange(len(methods))
    
    # F1-Score
    axes[0, 0].bar(x_pos, results_df['F1-Score'].values, color='skyblue', edgecolor='black')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].set_ylabel('F1-Score')
    axes[0, 0].set_title('F1-Score Comparison')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=results_df['F1-Score'].iloc[0], color='red', linestyle='--', label='Baseline')
    axes[0, 0].legend()
    
    # ROC-AUC
    axes[0, 1].bar(x_pos, results_df['ROC-AUC'].values, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].set_ylabel('ROC-AUC')
    axes[0, 1].set_title('ROC-AUC Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].axhline(y=results_df['ROC-AUC'].iloc[0], color='red', linestyle='--', label='Baseline')
    axes[0, 1].legend()
    
    # Sensitivity vs Specificity
    axes[1, 0].plot(x_pos, results_df['Sensitivity'].values, marker='o', label='Sensitivity (Recall)', linewidth=2)
    axes[1, 0].plot(x_pos, results_df['Specificity'].values, marker='s', label='Specificity', linewidth=2)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Sensitivity vs Specificity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Grouped bar chart
    width = 0.15
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Sensitivity', 'Precision']
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2) * width
        axes[1, 1].bar(x_pos + offset, results_df[metric].values, width, label=metric, edgecolor='black')
    
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task7_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task7_metrics_comparison.png")


def main():
    """Main function to demonstrate handling imbalanced data"""
    
    print("=" * 70)
    print("Task 7: Handling Imbalanced Data with Resampling Techniques")
    print("=" * 70)
    
    # Create imbalanced dataset
    df, X, y = create_imbalanced_dataset()
    
    print("\n" + "=" * 70)
    print("Dataset Overview:")
    print("=" * 70)
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Compare all resampling methods
    results_df, resampled_data = compare_resampling_methods(X, y)
    
    # Visualize confusion matrices
    visualize_confusion_matrices(results_df)
    
    # Visualize metrics comparison
    visualize_metrics_comparison(results_df)
    
    # Save results
    results_df.to_csv('task7_resampling_results.csv', index=False)
    print(f"\nSaved results to: task7_resampling_results.csv")
    
    # Analysis and recommendations
    print("\n" + "=" * 70)
    print("Analysis and Recommendations:")
    print("=" * 70)
    
    best_f1_idx = results_df['F1-Score'].idxmax()
    best_method = results_df.loc[best_f1_idx, 'Method']
    
    print(f"\nBest method by F1-Score: {best_method}")
    print(f"  F1-Score: {results_df.loc[best_f1_idx, 'F1-Score']:.4f}")
    print(f"  ROC-AUC: {results_df.loc[best_f1_idx, 'ROC-AUC']:.4f}")
    print(f"  Sensitivity: {results_df.loc[best_f1_idx, 'Sensitivity']:.4f}")
    
    baseline_f1 = results_df.loc[0, 'F1-Score']
    improvement = (results_df.loc[best_f1_idx, 'F1-Score'] - baseline_f1) / baseline_f1 * 100
    print(f"\nImprovement over baseline: {improvement:.1f}%")
    
    print("\n" + "=" * 70)
    print("Method Characteristics:")
    print("=" * 70)
    
    print("\n1. Random Oversampling:")
    print("   ✓ Simple and fast")
    print("   ✓ Preserves all original data")
    print("   ✗ May lead to overfitting (exact duplicates)")
    print("   ✗ Increases dataset size")
    
    print("\n2. Random Undersampling:")
    print("   ✓ Fast and reduces dataset size")
    print("   ✓ No synthetic data")
    print("   ✗ Loses potentially useful information")
    print("   ✗ May underfit if minority class is very small")
    
    print("\n3. SMOTE:")
    print("   ✓ Creates synthetic realistic samples")
    print("   ✓ Reduces overfitting compared to random oversampling")
    print("   ✓ Generally better performance")
    print("   ✗ May create noisy samples in overlapping regions")
    
    print("\n4. SMOTE + Tomek Links:")
    print("   ✓ Cleans decision boundary")
    print("   ✓ Removes ambiguous samples")
    print("   ✓ Better class separation")
    print("   ✗ Slightly more complex")
    
    print("\n5. SMOTE + ENN:")
    print("   ✓ More aggressive cleaning than Tomek")
    print("   ✓ Improves sample quality")
    print("   ✓ Often best for noisy data")
    print("   ✗ May remove too many samples")
    
    print("\n" + "=" * 70)
    print("Task 7 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Imbalanced data biases models toward the majority class")
    print("2. Random oversampling duplicates minority samples (risk of overfitting)")
    print("3. Random undersampling removes majority samples (risk of information loss)")
    print("4. SMOTE generates synthetic minority samples via interpolation")
    print("5. Combination methods (SMOTE + cleaning) often perform best")
    print("6. Use F1-score and ROC-AUC for imbalanced data (not just accuracy)")
    print("7. Monitor both sensitivity (minority recall) and specificity")
    print("8. Choose method based on dataset size, noise level, and domain requirements")


if __name__ == "__main__":
    main()
