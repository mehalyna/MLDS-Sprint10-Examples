"""
Task 5: Multi-Class Label Encoding with Ordinal Relationships
Demonstrates ordinal encoding while preserving natural order and handling new categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data():
    """Create a sample dataset with ordinal categorical features"""
    n_samples = 500
    
    # Education level (ordinal)
    education_levels = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD']
    education_weights = [0.35, 0.40, 0.20, 0.05]
    education = np.random.choice(education_levels, n_samples, p=education_weights)
    
    # Experience level (ordinal)
    experience_levels = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
    experience_weights = [0.15, 0.25, 0.30, 0.20, 0.10]
    experience = np.random.choice(experience_levels, n_samples, p=experience_weights)
    
    # Performance rating (ordinal)
    performance_ratings = ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
    performance_weights = [0.05, 0.15, 0.40, 0.30, 0.10]
    performance = np.random.choice(performance_ratings, n_samples, p=performance_weights)
    
    # Credit rating (ordinal)
    credit_ratings = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    credit_weights = [0.10, 0.20, 0.35, 0.25, 0.10]
    credit_rating = np.random.choice(credit_ratings, n_samples, p=credit_weights)
    
    # T-shirt size (ordinal)
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    size_weights = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]
    size = np.random.choice(sizes, n_samples, p=size_weights)
    
    # Priority level (ordinal)
    priorities = ['Low', 'Medium', 'High', 'Critical']
    priority_weights = [0.30, 0.40, 0.20, 0.10]
    priority = np.random.choice(priorities, n_samples, p=priority_weights)
    
    # Generate salary based on education and experience (target variable)
    edu_salary_map = {'High School': 40000, 'Bachelor\'s': 60000, 'Master\'s': 80000, 'PhD': 100000}
    exp_salary_map = {'Entry': 0, 'Junior': 10000, 'Mid': 25000, 'Senior': 40000, 'Expert': 60000}
    
    salary = np.array([edu_salary_map[e] + exp_salary_map[ex] + np.random.normal(0, 5000) 
                      for e, ex in zip(education, experience)])
    
    # Create salary category (for classification task)
    salary_category = pd.cut(salary, bins=[0, 50000, 70000, 90000, np.inf], 
                             labels=['Low', 'Medium', 'High', 'Very High'])
    
    df = pd.DataFrame({
        'education': education,
        'experience': experience,
        'performance': performance,
        'credit_rating': credit_rating,
        'size': size,
        'priority': priority,
        'age': np.random.randint(22, 65, n_samples),
        'years_of_service': np.random.randint(0, 30, n_samples),
        'salary': salary,
        'salary_category': salary_category
    })
    
    return df


def manual_ordinal_encoding(df, column, order):
    """
    Manually encode ordinal features with specified order
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to encode
    order : list
        Ordered list of categories from lowest to highest
        
    Returns:
    --------
    DataFrame with encoded column and mapping dictionary
    """
    df_encoded = df.copy()
    
    # Create mapping dictionary
    mapping = {category: idx for idx, category in enumerate(order)}
    
    # Apply encoding
    encoded_column_name = f'{column}_encoded'
    df_encoded[encoded_column_name] = df[column].map(mapping)
    
    # Handle missing values (categories not in the order)
    df_encoded[encoded_column_name] = df_encoded[encoded_column_name].fillna(-1)
    
    print(f"\nManual Ordinal Encoding for '{column}':")
    print(f"  Order: {order}")
    print(f"  Mapping: {mapping}")
    print(f"  Encoded column: {encoded_column_name}")
    
    # Show distribution
    print(f"\n  Original distribution:")
    print(f"{df[column].value_counts().sort_index()}")
    print(f"\n  Encoded distribution:")
    print(f"{df_encoded[encoded_column_name].value_counts().sort_index()}")
    
    return df_encoded, mapping


def sklearn_ordinal_encoding(df, columns_with_orders):
    """
    Use sklearn's OrdinalEncoder to encode multiple ordinal features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    columns_with_orders : dict
        Dictionary mapping column names to their ordered categories
        
    Returns:
    --------
    DataFrame with encoded columns and encoder object
    """
    print("\nsklearn Ordinal Encoding:")
    print("=" * 70)
    
    df_encoded = df.copy()
    
    # Prepare data for encoding
    columns = list(columns_with_orders.keys())
    categories = [columns_with_orders[col] for col in columns]
    
    # Create encoder
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Fit and transform
    encoded_values = encoder.fit_transform(df[columns])
    
    # Create new column names
    for idx, col in enumerate(columns):
        encoded_col_name = f'{col}_encoded'
        df_encoded[encoded_col_name] = encoded_values[:, idx]
        
        print(f"\nEncoded '{col}':")
        print(f"  Categories: {columns_with_orders[col]}")
        print(f"  Encoded range: [{df_encoded[encoded_col_name].min():.0f}, {df_encoded[encoded_col_name].max():.0f}]")
    
    return df_encoded, encoder


def handle_unknown_categories(df_train, df_test, column, order, default_value=-1):
    """
    Handle categories in test set that don't appear in training set
    
    Parameters:
    -----------
    df_train : DataFrame
        Training dataframe
    df_test : DataFrame
        Test dataframe
    column : str
        Column name to encode
    order : list
        Ordered list of categories
    default_value : int
        Value to assign to unknown categories
        
    Returns:
    --------
    Encoded training and test dataframes
    """
    print(f"\nHandling Unknown Categories for '{column}':")
    print("=" * 70)
    
    # Create mapping from training data
    mapping = {category: idx for idx, category in enumerate(order)}
    
    # Encode training data
    df_train_encoded = df_train.copy()
    encoded_col = f'{column}_encoded'
    df_train_encoded[encoded_col] = df_train[column].map(mapping)
    
    # Check for missing values in training
    train_missing = df_train_encoded[encoded_col].isna().sum()
    if train_missing > 0:
        print(f"  Warning: {train_missing} unknown categories in training data")
        df_train_encoded[encoded_col] = df_train_encoded[encoded_col].fillna(default_value)
    
    # Encode test data
    df_test_encoded = df_test.copy()
    df_test_encoded[encoded_col] = df_test[column].map(mapping)
    
    # Handle unknown categories in test set
    test_unknown = df_test_encoded[encoded_col].isna().sum()
    if test_unknown > 0:
        unknown_cats = df_test[df_test_encoded[encoded_col].isna()][column].unique()
        print(f"  Found {test_unknown} unknown categories in test set: {unknown_cats}")
        print(f"  Assigning default value: {default_value}")
        df_test_encoded[encoded_col] = df_test_encoded[encoded_col].fillna(default_value)
    else:
        print(f"  No unknown categories found in test set")
    
    print(f"\n  Training set encoding range: [{df_train_encoded[encoded_col].min():.0f}, {df_train_encoded[encoded_col].max():.0f}]")
    print(f"  Test set encoding range: [{df_test_encoded[encoded_col].min():.0f}, {df_test_encoded[encoded_col].max():.0f}]")
    
    return df_train_encoded, df_test_encoded, mapping


def validate_ordinal_encoding(df, original_col, encoded_col):
    """
    Validate that ordinal encoding preserves the order
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe with original and encoded columns
    original_col : str
        Original column name
    encoded_col : str
        Encoded column name
    """
    print(f"\nValidating Ordinal Encoding for '{original_col}':")
    print("=" * 70)
    
    # Check monotonicity
    grouped = df.groupby(original_col)[encoded_col].first().sort_index()
    
    print(f"\n  Category to Encoding mapping:")
    for cat, enc in grouped.items():
        print(f"    {cat:20s} -> {enc:.0f}")
    
    # Verify monotonic relationship
    encoded_values = grouped.values
    is_monotonic = all(encoded_values[i] <= encoded_values[i+1] 
                       for i in range(len(encoded_values)-1))
    
    if is_monotonic:
        print(f"\n  ✓ Encoding preserves ordinal relationship (monotonic)")
    else:
        print(f"\n  ✗ Warning: Encoding does NOT preserve ordinal relationship")
    
    # Check for gaps or duplicates
    unique_encoded = sorted(df[encoded_col].dropna().unique())
    expected_range = list(range(int(min(unique_encoded)), int(max(unique_encoded)) + 1))
    
    if unique_encoded == expected_range:
        print(f"  ✓ Encoding is continuous with no gaps")
    else:
        missing = set(expected_range) - set(unique_encoded)
        if missing:
            print(f"  ⚠ Gaps in encoding: {missing}")


def demonstrate_model_performance(df):
    """
    Demonstrate the importance of ordinal encoding in model performance
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe with features and target
    """
    print("\n" + "=" * 70)
    print("Model Performance Comparison: Ordinal vs One-Hot Encoding")
    print("=" * 70)
    
    # Define ordinal features and their orders
    ordinal_features = {
        'education': ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'],
        'experience': ['Entry', 'Junior', 'Mid', 'Senior', 'Expert'],
        'performance': ['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
    }
    
    # Prepare data
    target = 'salary_category'
    
    # Ordinal encoding
    df_ordinal = df.copy()
    for col, order in ordinal_features.items():
        mapping = {cat: idx for idx, cat in enumerate(order)}
        df_ordinal[f'{col}_encoded'] = df_ordinal[col].map(mapping)
    
    ordinal_cols = [f'{col}_encoded' for col in ordinal_features.keys()]
    X_ordinal = df_ordinal[ordinal_cols + ['age', 'years_of_service']]
    y = df[target]
    
    # One-hot encoding
    df_onehot = pd.get_dummies(df[list(ordinal_features.keys())], prefix=list(ordinal_features.keys()))
    X_onehot = pd.concat([df_onehot, df[['age', 'years_of_service']]], axis=1)
    
    # Split data
    X_train_ord, X_test_ord, y_train, y_test = train_test_split(
        X_ordinal, y, test_size=0.2, random_state=42
    )
    X_train_oh, X_test_oh, _, _ = train_test_split(
        X_onehot, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\nTraining Random Forest with Ordinal Encoding...")
    model_ordinal = RandomForestClassifier(n_estimators=100, random_state=42)
    model_ordinal.fit(X_train_ord, y_train)
    y_pred_ord = model_ordinal.predict(X_test_ord)
    acc_ordinal = accuracy_score(y_test, y_pred_ord)
    
    print("Training Random Forest with One-Hot Encoding...")
    model_onehot = RandomForestClassifier(n_estimators=100, random_state=42)
    model_onehot.fit(X_train_oh, y_train)
    y_pred_oh = model_onehot.predict(X_test_oh)
    acc_onehot = accuracy_score(y_test, y_pred_oh)
    
    # Comparison
    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Encoding Method': ['Ordinal Encoding', 'One-Hot Encoding'],
        'Number of Features': [len(X_ordinal.columns), len(X_onehot.columns)],
        'Accuracy': [acc_ordinal, acc_onehot]
    })
    
    print("\n", comparison.to_string(index=False))
    
    print(f"\nFeature count reduction: {len(X_onehot.columns)} → {len(X_ordinal.columns)} "
          f"({(1 - len(X_ordinal.columns)/len(X_onehot.columns))*100:.1f}% reduction)")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("Feature Importance (Ordinal Encoding):")
    print("=" * 70)
    
    importance_df = pd.DataFrame({
        'Feature': X_ordinal.columns,
        'Importance': model_ordinal.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n", importance_df.to_string(index=False))
    
    return acc_ordinal, acc_onehot


def visualize_ordinal_encoding(df):
    """Visualize ordinal encoding relationships"""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # Encode education
    edu_order = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD']
    edu_mapping = {cat: idx for idx, cat in enumerate(edu_order)}
    df['education_encoded'] = df['education'].map(edu_mapping)
    
    # Encode experience
    exp_order = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
    exp_mapping = {cat: idx for idx, cat in enumerate(exp_order)}
    df['experience_encoded'] = df['experience'].map(exp_mapping)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Education encoding
    edu_counts = df.groupby('education')['education_encoded'].first().sort_values()
    axes[0, 0].bar(range(len(edu_counts)), [df[df['education']==cat].shape[0] for cat in edu_counts.index])
    axes[0, 0].set_xticks(range(len(edu_counts)))
    axes[0, 0].set_xticklabels([f"{cat}\n({int(enc)})" for cat, enc in edu_counts.items()], rotation=45, ha='right')
    axes[0, 0].set_title('Education Level - Ordinal Encoding')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Experience encoding
    exp_counts = df.groupby('experience')['experience_encoded'].first().sort_values()
    axes[0, 1].bar(range(len(exp_counts)), [df[df['experience']==cat].shape[0] for cat in exp_counts.index], color='orange')
    axes[0, 1].set_xticks(range(len(exp_counts)))
    axes[0, 1].set_xticklabels([f"{cat}\n({int(enc)})" for cat, enc in exp_counts.items()], rotation=45, ha='right')
    axes[0, 1].set_title('Experience Level - Ordinal Encoding')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Salary by education (showing ordinal relationship)
    edu_salary = df.groupby('education')['salary'].mean().reindex(edu_order)
    axes[1, 0].plot(range(len(edu_salary)), edu_salary.values, marker='o', linewidth=2, markersize=10, color='green')
    axes[1, 0].set_xticks(range(len(edu_salary)))
    axes[1, 0].set_xticklabels(edu_salary.index, rotation=45, ha='right')
    axes[1, 0].set_title('Average Salary by Education Level\n(Shows Natural Order)')
    axes[1, 0].set_ylabel('Average Salary ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Heatmap of encoded values
    encoded_cols = ['education_encoded', 'experience_encoded']
    corr_matrix = df[encoded_cols + ['salary']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], 
                xticklabels=['Education', 'Experience', 'Salary'],
                yticklabels=['Education', 'Experience', 'Salary'])
    axes[1, 1].set_title('Correlation Matrix\n(Ordinal Encoded Features)')
    
    plt.tight_layout()
    plt.savefig('task5_ordinal_encoding.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task5_ordinal_encoding.png")


def main():
    """Main function to demonstrate ordinal encoding"""
    
    print("=" * 70)
    print("Task 5: Multi-Class Label Encoding with Ordinal Relationships")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("PART 1: Manual Ordinal Encoding")
    print("=" * 70)
    
    # Manual encoding for education
    education_order = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD']
    df, edu_mapping = manual_ordinal_encoding(df, 'education', education_order)
    
    # Manual encoding for experience
    experience_order = ['Entry', 'Junior', 'Mid', 'Senior', 'Expert']
    df, exp_mapping = manual_ordinal_encoding(df, 'experience', experience_order)
    
    print("\n" + "=" * 70)
    print("PART 2: sklearn Ordinal Encoding")
    print("=" * 70)
    
    # sklearn encoding for multiple features
    ordinal_features = {
        'performance': ['Poor', 'Below Average', 'Average', 'Good', 'Excellent'],
        'credit_rating': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
        'size': ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
        'priority': ['Low', 'Medium', 'High', 'Critical']
    }
    
    df, encoder = sklearn_ordinal_encoding(df, ordinal_features)
    
    print("\n" + "=" * 70)
    print("PART 3: Handling Unknown Categories")
    print("=" * 70)
    
    # Split data to simulate train/test with unknown categories
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)
    
    # Artificially introduce unknown category
    df_test_modified = df_test.copy()
    # Add some unknown experience levels
    df_test_modified.loc[df_test_modified.index[0:5], 'experience'] = 'Consultant'
    
    df_train_enc, df_test_enc, mapping = handle_unknown_categories(
        df_train, df_test_modified, 'experience', experience_order, default_value=-1
    )
    
    print("\n" + "=" * 70)
    print("PART 4: Validation of Ordinal Encoding")
    print("=" * 70)
    
    # Validate encodings
    validate_ordinal_encoding(df, 'education', 'education_encoded')
    validate_ordinal_encoding(df, 'experience', 'experience_encoded')
    validate_ordinal_encoding(df, 'performance', 'performance_encoded')
    
    print("\n" + "=" * 70)
    print("PART 5: Model Performance Comparison")
    print("=" * 70)
    
    # Demonstrate model performance
    acc_ordinal, acc_onehot = demonstrate_model_performance(df)
    
    print("\n" + "=" * 70)
    print("PART 6: Visualizations")
    print("=" * 70)
    
    # Create visualizations
    visualize_ordinal_encoding(df)
    
    # Save encoded dataset
    df.to_csv('task5_ordinal_encoded.csv', index=False)
    print(f"\nSaved encoded dataset to: task5_ordinal_encoded.csv")
    
    print("\n" + "=" * 70)
    print("Task 5 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Ordinal encoding preserves natural order of categories (e.g., Low < Medium < High)")
    print("2. Significantly reduces feature dimensionality compared to one-hot encoding")
    print("3. Works well with tree-based models that can interpret ordering")
    print("4. Handle unknown categories with a default value (e.g., -1)")
    print("5. Always validate that encoding preserves the intended order")
    print("6. More interpretable than one-hot encoding for ordered categories")
    print("7. Can capture linear relationships between ordinal features and target")
    print("8. Essential for features like education, experience, ratings, sizes, etc.")


if __name__ == "__main__":
    main()
