"""
Task 3: Advanced Data Imputation with Multiple Techniques
Demonstrates KNN, Iterative (MICE), Forward/Backward Fill, and Constant Value imputation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data_with_missing():
    """Create a sample dataset with intentional missing values"""
    n_samples = 200
    
    # Generate base data
    age = np.random.normal(35, 10, n_samples)
    income = age * 1500 + np.random.normal(0, 10000, n_samples)  # Correlated with age
    experience = age * 0.4 - 8 + np.random.normal(0, 2, n_samples)  # Correlated with age
    credit_score = 650 + income * 0.002 + np.random.normal(0, 50, n_samples)  # Correlated with income
    satisfaction = np.random.randint(1, 6, n_samples)  # 1-5 rating scale
    
    # Create time-series data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'age': age,
        'income': income,
        'experience': experience,
        'credit_score': credit_score,
        'satisfaction': satisfaction,
        'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], n_samples),
        'project_count': np.random.randint(0, 10, n_samples)
    })
    
    # Introduce missing values with different patterns
    # Random missing values (MCAR - Missing Completely At Random)
    missing_indices_age = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    df.loc[missing_indices_age, 'age'] = np.nan
    
    missing_indices_income = np.random.choice(n_samples, size=int(n_samples * 0.20), replace=False)
    df.loc[missing_indices_income, 'income'] = np.nan
    
    missing_indices_exp = np.random.choice(n_samples, size=int(n_samples * 0.12), replace=False)
    df.loc[missing_indices_exp, 'experience'] = np.nan
    
    missing_indices_credit = np.random.choice(n_samples, size=int(n_samples * 0.18), replace=False)
    df.loc[missing_indices_credit, 'credit_score'] = np.nan
    
    missing_indices_dept = np.random.choice(n_samples, size=int(n_samples * 0.10), replace=False)
    df.loc[missing_indices_dept, 'department'] = np.nan
    
    missing_indices_proj = np.random.choice(n_samples, size=int(n_samples * 0.08), replace=False)
    df.loc[missing_indices_proj, 'project_count'] = np.nan
    
    return df


def analyze_missing_data(df):
    """Analyze and visualize missing data patterns"""
    print("\nMissing Data Analysis:")
    print("=" * 70)
    
    missing_stats = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Data_Type': df.dtypes
    })
    missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    print("\nMissing Values Summary:")
    print(missing_stats)
    
    return missing_stats


def knn_imputation(df, columns, n_neighbors=5):
    """
    Impute missing values using K-Nearest Neighbors
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to impute
    n_neighbors : int
        Number of neighbors to use
        
    Returns:
    --------
    DataFrame with KNN imputed values
    """
    print(f"\nKNN Imputation (k={n_neighbors}):")
    print("=" * 70)
    
    df_knn = df.copy()
    
    # KNN imputer works only with numerical data
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Select only numerical columns
    numerical_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        df_knn[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        print(f"Imputed columns: {numerical_cols}")
        print(f"Method: KNN with {n_neighbors} nearest neighbors")
        print("How it works: Uses weighted average of k-nearest neighbors")
        
        # Show before/after comparison
        for col in numerical_cols[:2]:  # Show first 2 columns
            original_missing = df[col].isnull().sum()
            if original_missing > 0:
                print(f"\n  {col}:")
                print(f"    Original missing: {original_missing}")
                print(f"    After imputation: {df_knn[col].isnull().sum()}")
                print(f"    Mean (original): {df[col].mean():.2f}")
                print(f"    Mean (imputed): {df_knn[col].mean():.2f}")
    
    return df_knn


def iterative_imputation(df, columns, max_iter=10):
    """
    Impute missing values using Iterative Imputation (MICE)
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to impute
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    DataFrame with iteratively imputed values
    """
    print(f"\nIterative Imputation (MICE - Multiple Imputation by Chained Equations):")
    print("=" * 70)
    
    df_mice = df.copy()
    
    # Select only numerical columns
    numerical_cols = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        # Use RandomForest as the estimator for better performance
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=max_iter,
            random_state=42,
            verbose=0
        )
        
        df_mice[numerical_cols] = imputer.fit_transform(df[numerical_cols])
        
        print(f"Imputed columns: {numerical_cols}")
        print(f"Method: Iterative imputation with Random Forest estimator")
        print(f"Max iterations: {max_iter}")
        print("How it works: Models each feature as a function of other features iteratively")
        
        # Show before/after comparison
        for col in numerical_cols[:2]:  # Show first 2 columns
            original_missing = df[col].isnull().sum()
            if original_missing > 0:
                print(f"\n  {col}:")
                print(f"    Original missing: {original_missing}")
                print(f"    After imputation: {df_mice[col].isnull().sum()}")
                print(f"    Mean (original): {df[col].mean():.2f}")
                print(f"    Mean (imputed): {df_mice[col].mean():.2f}")
    
    return df_mice


def forward_backward_fill(df, columns):
    """
    Impute missing values using forward fill and backward fill
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe (should be sorted by date/time)
    columns : list
        List of column names to impute
        
    Returns:
    --------
    Two DataFrames: one with forward fill, one with backward fill
    """
    print("\nForward Fill and Backward Fill (for time-series data):")
    print("=" * 70)
    
    # Ensure data is sorted by date
    df_sorted = df.sort_values('date').copy()
    
    # Forward fill
    df_ffill = df_sorted.copy()
    df_ffill[columns] = df_ffill[columns].fillna(method='ffill')
    
    # Backward fill
    df_bfill = df_sorted.copy()
    df_bfill[columns] = df_bfill[columns].fillna(method='bfill')
    
    # Combined approach: forward fill then backward fill
    df_combined = df_sorted.copy()
    df_combined[columns] = df_combined[columns].fillna(method='ffill').fillna(method='bfill')
    
    print("Method: Propagate valid observations forward/backward")
    print("Use case: Time-series data where temporal consistency is important")
    
    for col in columns[:2]:  # Show first 2 columns
        original_missing = df[col].isnull().sum()
        if original_missing > 0:
            print(f"\n  {col}:")
            print(f"    Original missing: {original_missing}")
            print(f"    After forward fill: {df_ffill[col].isnull().sum()}")
            print(f"    After backward fill: {df_bfill[col].isnull().sum()}")
            print(f"    After combined (ffill+bfill): {df_combined[col].isnull().sum()}")
    
    return df_ffill, df_bfill, df_combined


def constant_value_imputation(df, impute_dict):
    """
    Impute missing values with constant values
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    impute_dict : dict
        Dictionary mapping column names to constant values
        
    Returns:
    --------
    DataFrame with constant value imputed
    """
    print("\nConstant Value Imputation:")
    print("=" * 70)
    
    df_constant = df.copy()
    
    print("Method: Fill with domain-specific constant values")
    print("\nImputation mapping:")
    
    for col, value in impute_dict.items():
        if col in df_constant.columns:
            original_missing = df_constant[col].isnull().sum()
            df_constant[col] = df_constant[col].fillna(value)
            
            print(f"  {col}: '{value}' ({original_missing} values imputed)")
    
    return df_constant


def visualize_imputation_comparison(df_original, df_knn, df_mice, column):
    """
    Visualize the effect of different imputation methods
    
    Parameters:
    -----------
    df_original : DataFrame
        Original dataframe with missing values
    df_knn : DataFrame
        Dataframe with KNN imputation
    df_mice : DataFrame
        Dataframe with MICE imputation
    column : str
        Column to visualize
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original distribution (non-missing values only)
    axes[0, 0].hist(df_original[column].dropna(), bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_title(f'Original {column} (non-missing only)')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df_original[column].mean(), color='red', linestyle='--', label=f'Mean: {df_original[column].mean():.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KNN imputed
    axes[0, 1].hist(df_knn[column], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title(f'KNN Imputation')
    axes[0, 1].set_xlabel(column)
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df_knn[column].mean(), color='red', linestyle='--', label=f'Mean: {df_knn[column].mean():.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MICE imputed
    axes[1, 0].hist(df_mice[column], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title(f'Iterative (MICE) Imputation')
    axes[1, 0].set_xlabel(column)
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(df_mice[column].mean(), color='red', linestyle='--', label=f'Mean: {df_mice[column].mean():.2f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    data_to_plot = [
        df_original[column].dropna(),
        df_knn[column],
        df_mice[column]
    ]
    axes[1, 1].boxplot(data_to_plot, labels=['Original\n(non-missing)', 'KNN', 'MICE'])
    axes[1, 1].set_title('Comparison of Methods')
    axes[1, 1].set_ylabel(column)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'task3_imputation_{column}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization: task3_imputation_{column}.png")


def compare_imputation_methods(df_original, df_knn, df_mice, columns):
    """Compare statistics across different imputation methods"""
    print("\n" + "=" * 70)
    print("Comparison of Imputation Methods")
    print("=" * 70)
    
    comparison_results = []
    
    for col in columns:
        if df_original[col].dtype in [np.float64, np.int64]:
            result = {
                'Column': col,
                'Original_Missing': df_original[col].isnull().sum(),
                'Original_Mean': df_original[col].mean(),
                'Original_Std': df_original[col].std(),
                'KNN_Mean': df_knn[col].mean(),
                'KNN_Std': df_knn[col].std(),
                'MICE_Mean': df_mice[col].mean(),
                'MICE_Std': df_mice[col].std()
            }
            comparison_results.append(result)
    
    comparison_df = pd.DataFrame(comparison_results)
    print("\nStatistical Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """Main function to demonstrate advanced imputation techniques"""
    
    print("=" * 70)
    print("Task 3: Advanced Data Imputation with Multiple Techniques")
    print("=" * 70)
    
    # Create sample data with missing values
    df_original = create_sample_data_with_missing()
    print(f"\nOriginal dataset shape: {df_original.shape}")
    print("\nFirst few rows:")
    print(df_original.head(10))
    
    # Analyze missing data
    missing_stats = analyze_missing_data(df_original)
    
    print("\n" + "=" * 70)
    print("METHOD 1: K-Nearest Neighbors (KNN) Imputation")
    print("=" * 70)
    
    # Apply KNN imputation
    numerical_cols = ['age', 'income', 'experience', 'credit_score', 'project_count']
    df_knn = knn_imputation(df_original, numerical_cols, n_neighbors=5)
    
    print("\n" + "=" * 70)
    print("METHOD 2: Iterative Imputation (MICE)")
    print("=" * 70)
    
    # Apply MICE imputation
    df_mice = iterative_imputation(df_original, numerical_cols, max_iter=10)
    
    print("\n" + "=" * 70)
    print("METHOD 3: Forward Fill and Backward Fill")
    print("=" * 70)
    
    # Apply forward/backward fill
    time_series_cols = ['age', 'income', 'experience', 'project_count']
    df_ffill, df_bfill, df_combined = forward_backward_fill(df_original, time_series_cols)
    
    print("\n" + "=" * 70)
    print("METHOD 4: Constant Value Imputation")
    print("=" * 70)
    
    # Apply constant value imputation
    impute_values = {
        'department': 'Unknown',
        'project_count': 0,
        'satisfaction': 3  # Neutral rating
    }
    df_constant = constant_value_imputation(df_original, impute_values)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION: Compare Imputation Methods")
    print("=" * 70)
    
    # Visualize imputation results
    for col in ['age', 'income']:
        visualize_imputation_comparison(df_original, df_knn, df_mice, col)
    
    print("\n" + "=" * 70)
    print("COMPARISON: Statistical Analysis")
    print("=" * 70)
    
    # Compare methods
    comparison_df = compare_imputation_methods(df_original, df_knn, df_mice, numerical_cols)
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("\nWhen to use each method:")
    print("\n1. KNN Imputation:")
    print("   - Best for: Data with local patterns and similar observations")
    print("   - Pros: Considers feature relationships, handles multivariate patterns")
    print("   - Cons: Sensitive to outliers, computationally expensive for large datasets")
    
    print("\n2. Iterative Imputation (MICE):")
    print("   - Best for: Complex relationships between features")
    print("   - Pros: Captures complex patterns, handles multiple variables simultaneously")
    print("   - Cons: Computationally expensive, requires many iterations")
    
    print("\n3. Forward/Backward Fill:")
    print("   - Best for: Time-series data with temporal dependencies")
    print("   - Pros: Maintains temporal consistency, fast")
    print("   - Cons: Not suitable for non-temporal data, can propagate errors")
    
    print("\n4. Constant Value Imputation:")
    print("   - Best for: Categorical data, domain-specific scenarios")
    print("   - Pros: Simple, interpretable, fast")
    print("   - Cons: Doesn't capture relationships, may introduce bias")
    
    # Save results
    df_knn.to_csv('task3_knn_imputed.csv', index=False)
    df_mice.to_csv('task3_mice_imputed.csv', index=False)
    df_combined.to_csv('task3_ffill_bfill_imputed.csv', index=False)
    print(f"\nSaved imputed datasets to CSV files")
    
    print("\n" + "=" * 70)
    print("Task 3 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. KNN imputation leverages feature similarity for accurate imputation")
    print("2. MICE captures complex multivariate relationships through iteration")
    print("3. Forward/backward fill preserves temporal patterns in time-series")
    print("4. Constant value imputation provides simple, interpretable results")
    print("5. Method choice depends on data characteristics and missing patterns")
    print("6. Always validate imputation quality by comparing statistics")


if __name__ == "__main__":
    main()
