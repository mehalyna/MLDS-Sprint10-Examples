"""
Task 2: Feature Binning and Discretization
Demonstrates equal-width, equal-frequency, and custom binning techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample dataset
def create_sample_data():
    """Create a sample dataset with continuous variables"""
    n_samples = 500
    
    # Generate realistic data
    age = np.random.randint(18, 80, n_samples)
    income = np.random.lognormal(10.5, 0.8, n_samples)  # Skewed distribution
    hours_worked = np.clip(np.random.normal(40, 10, n_samples), 10, 70)
    credit_score = np.clip(np.random.normal(680, 100, n_samples), 300, 850)
    
    df = pd.DataFrame({
        'age': age,
        'income': income,
        'hours_worked': hours_worked,
        'credit_score': credit_score,
        'department': np.random.choice(['Sales', 'IT', 'HR', 'Finance', 'Marketing'], n_samples)
    })
    
    return df


def equal_width_binning(df, column, n_bins=5):
    """
    Divide the range of a continuous feature into equal-width bins
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to bin
    n_bins : int
        Number of bins to create
        
    Returns:
    --------
    Series with binned values and bin edges
    """
    binned_column = f'{column}_equal_width'
    
    # Create equal-width bins
    df[binned_column], bin_edges = pd.cut(
        df[column], 
        bins=n_bins, 
        labels=[f'Bin_{i+1}' for i in range(n_bins)],
        retbins=True,
        include_lowest=True
    )
    
    print(f"\nEqual-Width Binning for '{column}':")
    print(f"  Number of bins: {n_bins}")
    print(f"  Bin edges: {[f'{edge:.2f}' for edge in bin_edges]}")
    print(f"\n  Distribution:")
    print(df[binned_column].value_counts().sort_index())
    
    return df[binned_column], bin_edges


def equal_frequency_binning(df, column, n_bins=5):
    """
    Create bins such that each bin contains approximately the same number of observations
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to bin
    n_bins : int
        Number of bins to create
        
    Returns:
    --------
    Series with binned values and bin edges
    """
    binned_column = f'{column}_equal_freq'
    
    # Create equal-frequency bins using quantiles
    df[binned_column], bin_edges = pd.qcut(
        df[column], 
        q=n_bins, 
        labels=[f'Quantile_{i+1}' for i in range(n_bins)],
        retbins=True,
        duplicates='drop'
    )
    
    print(f"\nEqual-Frequency Binning for '{column}':")
    print(f"  Number of bins: {n_bins}")
    print(f"  Bin edges: {[f'{edge:.2f}' for edge in bin_edges]}")
    print(f"\n  Distribution:")
    print(df[binned_column].value_counts().sort_index())
    
    return df[binned_column], bin_edges


def custom_binning_age(df):
    """
    Apply custom binning for age based on life stages
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
        
    Returns:
    --------
    Series with custom age categories
    """
    bins = [0, 25, 35, 50, 65, 100]
    labels = ['Young Adult (18-25)', 'Adult (26-35)', 'Middle Age (36-50)', 
              'Senior (51-65)', 'Elderly (65+)']
    
    df['age_category'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
    
    print(f"\nCustom Binning for 'age' (Life Stages):")
    print(f"  Bins: {bins}")
    print(f"  Labels: {labels}")
    print(f"\n  Distribution:")
    print(df['age_category'].value_counts().sort_index())
    
    return df['age_category']


def custom_binning_income(df):
    """
    Apply custom binning for income based on economic classifications
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
        
    Returns:
    --------
    Series with custom income categories
    """
    bins = [0, 30000, 75000, 150000, np.inf]
    labels = ['Low Income (<$30k)', 'Middle Income ($30k-$75k)', 
              'Upper Middle ($75k-$150k)', 'High Income (>$150k)']
    
    df['income_category'] = pd.cut(df['income'], bins=bins, labels=labels, include_lowest=True)
    
    print(f"\nCustom Binning for 'income' (Economic Classes):")
    print(f"  Bins: {bins}")
    print(f"  Labels: {labels}")
    print(f"\n  Distribution:")
    print(df['income_category'].value_counts())
    
    return df['income_category']


def custom_binning_credit_score(df):
    """
    Apply custom binning for credit score based on credit rating standards
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
        
    Returns:
    --------
    Series with custom credit score categories
    """
    bins = [0, 580, 670, 740, 800, 850]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    
    df['credit_rating'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    
    print(f"\nCustom Binning for 'credit_score' (Credit Ratings):")
    print(f"  Bins: {bins}")
    print(f"  Labels: {labels}")
    print(f"\n  Distribution:")
    print(df['credit_rating'].value_counts())
    
    return df['credit_rating']


def visualize_binning(df, original_column, binned_columns, method_names):
    """
    Visualize the effect of different binning methods
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    original_column : str
        Original continuous column name
    binned_columns : list
        List of binned column names
    method_names : list
        List of method names for titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Original distribution
    axes[0].hist(df[original_column], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Original {original_column} Distribution')
    axes[0].set_xlabel(original_column)
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Binned distributions
    for idx, (binned_col, method) in enumerate(zip(binned_columns, method_names), 1):
        if binned_col in df.columns:
            counts = df[binned_col].value_counts().sort_index()
            axes[idx].bar(range(len(counts)), counts.values, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{method}')
            axes[idx].set_xlabel('Bins')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_xticks(range(len(counts)))
            axes[idx].set_xticklabels(counts.index, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'task2_binning_{original_column}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization: task2_binning_{original_column}.png")


def compare_binning_statistics(df, original_column, binned_columns):
    """
    Compare statistics across different binning methods
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    original_column : str
        Original continuous column
    binned_columns : list
        List of binned column names
    """
    print(f"\n{'='*70}")
    print(f"Statistics Comparison for '{original_column}'")
    print(f"{'='*70}")
    
    for binned_col in binned_columns:
        if binned_col in df.columns:
            print(f"\n{binned_col}:")
            grouped = df.groupby(binned_col)[original_column].agg(['mean', 'min', 'max', 'count'])
            print(grouped)


def main():
    """Main function to demonstrate feature binning and discretization"""
    
    print("=" * 70)
    print("Task 2: Feature Binning and Discretization")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("PART 1: Equal-Width Binning")
    print("=" * 70)
    
    # Apply equal-width binning to age
    equal_width_binning(df, 'age', n_bins=5)
    
    # Apply equal-width binning to hours_worked
    equal_width_binning(df, 'hours_worked', n_bins=4)
    
    print("\n" + "=" * 70)
    print("PART 2: Equal-Frequency Binning")
    print("=" * 70)
    
    # Apply equal-frequency binning to age
    equal_frequency_binning(df, 'age', n_bins=5)
    
    # Apply equal-frequency binning to income
    equal_frequency_binning(df, 'income', n_bins=5)
    
    print("\n" + "=" * 70)
    print("PART 3: Custom Binning Based on Domain Knowledge")
    print("=" * 70)
    
    # Apply custom binning
    custom_binning_age(df)
    custom_binning_income(df)
    custom_binning_credit_score(df)
    
    print("\n" + "=" * 70)
    print("PART 4: Visualize Binning Results")
    print("=" * 70)
    
    # Visualize age binning
    visualize_binning(
        df, 
        'age', 
        ['age_equal_width', 'age_equal_freq', 'age_category'],
        ['Equal-Width Binning', 'Equal-Frequency Binning', 'Custom Binning (Life Stages)']
    )
    
    # Visualize income binning
    visualize_binning(
        df,
        'income',
        ['income_equal_freq', 'income_category', 'income_equal_freq'],  # Reuse for layout
        ['Equal-Frequency Binning', 'Custom Binning (Economic Classes)', 'Equal-Frequency (Repeated)']
    )
    
    print("\n" + "=" * 70)
    print("PART 5: Compare Statistics Across Binning Methods")
    print("=" * 70)
    
    # Compare statistics for age
    compare_binning_statistics(df, 'age', ['age_equal_width', 'age_equal_freq', 'age_category'])
    
    # Compare statistics for income
    compare_binning_statistics(df, 'income', ['income_equal_freq', 'income_category'])
    
    print("\n" + "=" * 70)
    print("PART 6: Example Use Cases")
    print("=" * 70)
    
    print("\nExample 1: Crosstab of Age Category vs Department")
    crosstab = pd.crosstab(df['age_category'], df['department'])
    print(crosstab)
    
    print("\n\nExample 2: Average Income by Credit Rating")
    avg_income = df.groupby('credit_rating')['income'].mean().sort_values(ascending=False)
    print(avg_income)
    
    print("\n\nExample 3: Credit Score Distribution by Income Category")
    score_by_income = df.groupby('income_category')['credit_score'].describe()
    print(score_by_income)
    
    # Save the processed dataset
    output_columns = ['age', 'age_equal_width', 'age_equal_freq', 'age_category',
                     'income', 'income_equal_freq', 'income_category',
                     'hours_worked', 'hours_worked_equal_width',
                     'credit_score', 'credit_rating', 'department']
    df_output = df[output_columns]
    df_output.to_csv('task2_binned_data.csv', index=False)
    print(f"\nSaved processed data to: task2_binned_data.csv")
    
    print("\n" + "=" * 70)
    print("Task 2 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Equal-width binning creates uniform intervals but may result in unbalanced counts")
    print("2. Equal-frequency binning ensures balanced bins but intervals may vary greatly")
    print("3. Custom binning leverages domain knowledge for meaningful categories")
    print("4. Binning can improve model interpretability and handle non-linear relationships")
    print("5. Choice of binning method depends on data distribution and analysis goals")
    print("6. Categorical bins can be used for crosstabs, group analysis, and encoding")


if __name__ == "__main__":
    main()
