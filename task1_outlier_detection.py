"""
Task 1: Outlier Detection and Treatment
Demonstrates IQR-based outlier detection with capping and removal methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample dataset with outliers
def create_sample_data():
    """Create a sample dataset with intentional outliers"""
    n_samples = 200
    
    # Normal data
    age = np.random.normal(35, 10, n_samples)
    salary = np.random.normal(50000, 15000, n_samples)
    experience = np.random.normal(8, 3, n_samples)
    
    # Add outliers (5% of data)
    n_outliers = int(n_samples * 0.05)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    age[outlier_indices[:3]] = [80, 85, 90]  # Age outliers
    salary[outlier_indices[3:7]] = [150000, 180000, 200000, 250000]  # Salary outliers
    experience[outlier_indices[7:]] = [25, 30, 35]  # Experience outliers
    
    df = pd.DataFrame({
        'age': age,
        'salary': salary,
        'experience': experience,
        'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], n_samples)
    })
    
    return df


def detect_outliers_iqr(df, column):
    """
    Detect outliers using the IQR (Interquartile Range) method
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
        
    Returns:
    --------
    outliers : Series
        Boolean series indicating outlier positions
    lower_bound : float
        Lower boundary for outliers
    upper_bound : float
        Upper boundary for outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    print(f"\n{column} - Outlier Detection:")
    print(f"  Q1: {Q1:.2f}")
    print(f"  Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")
    print(f"  Number of outliers: {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")
    
    return outliers, lower_bound, upper_bound


def cap_outliers(df, column, lower_bound, upper_bound):
    """
    Cap outliers by replacing them with boundary values
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to cap outliers
    lower_bound : float
        Lower boundary
    upper_bound : float
        Upper boundary
        
    Returns:
    --------
    DataFrame with capped values
    """
    df_capped = df.copy()
    df_capped[column] = df_capped[column].clip(lower=lower_bound, upper=upper_bound)
    return df_capped


def remove_outliers(df, columns, threshold=0.05):
    """
    Remove rows containing outliers if they represent less than threshold of data
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    threshold : float
        Maximum percentage of outliers to remove (default: 5%)
        
    Returns:
    --------
    DataFrame with outlier rows removed
    """
    df_clean = df.copy()
    outlier_mask = pd.Series([False] * len(df))
    
    for column in columns:
        outliers, _, _ = detect_outliers_iqr(df, column)
        outlier_mask = outlier_mask | outliers
    
    total_outliers = outlier_mask.sum()
    outlier_percentage = total_outliers / len(df)
    
    print(f"\nTotal rows with outliers: {total_outliers} ({outlier_percentage*100:.2f}%)")
    
    if outlier_percentage <= threshold:
        df_clean = df_clean[~outlier_mask]
        print(f"Removed {total_outliers} rows with outliers")
    else:
        print(f"Warning: Outliers exceed {threshold*100}% threshold. Consider capping instead.")
        return df
    
    return df_clean


def visualize_outliers(df_original, df_capped, df_removed, column):
    """
    Visualize the effect of outlier treatment methods
    
    Parameters:
    -----------
    df_original : DataFrame
        Original dataframe
    df_capped : DataFrame
        Dataframe with capped outliers
    df_removed : DataFrame
        Dataframe with outliers removed
    column : str
        Column to visualize
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original data
    axes[0].boxplot(df_original[column].dropna())
    axes[0].set_title(f'Original {column}')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Capped data
    axes[1].boxplot(df_capped[column].dropna())
    axes[1].set_title(f'Capped {column}')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    # Removed outliers
    axes[2].boxplot(df_removed[column].dropna())
    axes[2].set_title(f'Outliers Removed {column}')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'task1_outliers_{column}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: task1_outliers_{column}.png")


def main():
    """Main function to demonstrate outlier detection and treatment"""
    
    print("=" * 70)
    print("Task 1: Outlier Detection and Treatment")
    print("=" * 70)
    
    # Create sample data
    df_original = create_sample_data()
    print(f"\nOriginal dataset shape: {df_original.shape}")
    print("\nFirst few rows:")
    print(df_original.head(10))
    
    print("\n" + "=" * 70)
    print("STEP 1: Detect Outliers Using IQR Method")
    print("=" * 70)
    
    # Detect outliers for numerical columns
    numerical_columns = ['age', 'salary', 'experience']
    outlier_info = {}
    
    for column in numerical_columns:
        outliers, lower, upper = detect_outliers_iqr(df_original, column)
        outlier_info[column] = (outliers, lower, upper)
    
    print("\n" + "=" * 70)
    print("STEP 2: Handle Outliers - Method 1 (Capping)")
    print("=" * 70)
    
    # Apply capping to all numerical columns
    df_capped = df_original.copy()
    for column in numerical_columns:
        _, lower, upper = outlier_info[column]
        df_capped = cap_outliers(df_capped, column, lower, upper)
        print(f"\nCapped {column}:")
        print(f"  Original range: [{df_original[column].min():.2f}, {df_original[column].max():.2f}]")
        print(f"  Capped range: [{df_capped[column].min():.2f}, {df_capped[column].max():.2f}]")
    
    print("\n" + "=" * 70)
    print("STEP 3: Handle Outliers - Method 2 (Removal)")
    print("=" * 70)
    
    # Remove outliers
    df_removed = remove_outliers(df_original, numerical_columns, threshold=0.05)
    print(f"\nResulting dataset shape: {df_removed.shape}")
    print(f"Rows removed: {len(df_original) - len(df_removed)}")
    
    print("\n" + "=" * 70)
    print("STEP 4: Compare Statistics")
    print("=" * 70)
    
    for column in numerical_columns:
        print(f"\n{column.upper()} Statistics:")
        print(f"  Original - Mean: {df_original[column].mean():.2f}, Std: {df_original[column].std():.2f}")
        print(f"  Capped   - Mean: {df_capped[column].mean():.2f}, Std: {df_capped[column].std():.2f}")
        print(f"  Removed  - Mean: {df_removed[column].mean():.2f}, Std: {df_removed[column].std():.2f}")
    
    print("\n" + "=" * 70)
    print("STEP 5: Generate Visualizations")
    print("=" * 70)
    
    # Create visualizations for each numerical column
    for column in numerical_columns:
        visualize_outliers(df_original, df_capped, df_removed, column)
    
    print("\n" + "=" * 70)
    print("Task 1 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. IQR method effectively identifies outliers using statistical boundaries")
    print("2. Capping preserves all data points while limiting extreme values")
    print("3. Removal eliminates outliers but reduces dataset size")
    print("4. Choice between methods depends on data context and analysis goals")
    print("5. Always visualize data before and after treatment to verify results")


if __name__ == "__main__":
    main()
