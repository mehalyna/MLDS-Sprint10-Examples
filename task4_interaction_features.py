"""
Task 4: Creating Interaction Features
Demonstrates multiplication interactions, division interactions, and polynomial features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data():
    """Create a sample dataset for demonstrating interaction features"""
    n_samples = 300
    
    # Generate base features
    height = np.random.uniform(150, 200, n_samples)  # cm
    width = np.random.uniform(50, 100, n_samples)    # cm
    depth = np.random.uniform(30, 80, n_samples)     # cm
    weight = np.random.uniform(10, 50, n_samples)    # kg
    
    # Generate features with relationships
    price = (height * width * depth / 10000) * 50 + weight * 10 + np.random.normal(0, 20, n_samples)
    production_time = (height + width + depth) / 10 + weight * 0.5 + np.random.normal(0, 2, n_samples)
    
    # Sales data
    units_sold = np.random.randint(10, 100, n_samples)
    revenue = units_sold * price + np.random.normal(0, 500, n_samples)
    
    # Store data
    store_size = np.random.uniform(500, 5000, n_samples)  # sq ft
    foot_traffic = np.random.uniform(100, 1000, n_samples)  # people per day
    
    df = pd.DataFrame({
        'height': height,
        'width': width,
        'depth': depth,
        'weight': weight,
        'price': price,
        'production_time': production_time,
        'units_sold': units_sold,
        'revenue': revenue,
        'store_size': store_size,
        'foot_traffic': foot_traffic
    })
    
    return df


def create_multiplication_interactions(df, feature_pairs):
    """
    Create interaction features by multiplying pairs of features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    feature_pairs : list of tuples
        List of feature pairs to multiply
        
    Returns:
    --------
    DataFrame with new multiplication interaction features
    """
    print("\nMultiplication Interactions:")
    print("=" * 70)
    print("Purpose: Capture combined effects of features")
    print("Example: height × width = area\n")
    
    df_result = df.copy()
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            new_feature_name = f'{feat1}_x_{feat2}'
            df_result[new_feature_name] = df[feat1] * df[feat2]
            
            print(f"Created: {new_feature_name}")
            print(f"  Formula: {feat1} × {feat2}")
            print(f"  Range: [{df_result[new_feature_name].min():.2f}, {df_result[new_feature_name].max():.2f}]")
            print(f"  Mean: {df_result[new_feature_name].mean():.2f}\n")
    
    return df_result


def create_division_interactions(df, feature_pairs):
    """
    Create ratio features by dividing one feature by another
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    feature_pairs : list of tuples
        List of (numerator, denominator) feature pairs
        
    Returns:
    --------
    DataFrame with new division interaction features
    """
    print("\nDivision Interactions (Ratios):")
    print("=" * 70)
    print("Purpose: Capture relative proportions and rates")
    print("Example: price / volume = price_per_unit_volume\n")
    
    df_result = df.copy()
    
    for numerator, denominator in feature_pairs:
        if numerator in df.columns and denominator in df.columns:
            new_feature_name = f'{numerator}_per_{denominator}'
            # Avoid division by zero
            df_result[new_feature_name] = df[numerator] / (df[denominator] + 1e-10)
            
            print(f"Created: {new_feature_name}")
            print(f"  Formula: {numerator} ÷ {denominator}")
            print(f"  Range: [{df_result[new_feature_name].min():.2f}, {df_result[new_feature_name].max():.2f}]")
            print(f"  Mean: {df_result[new_feature_name].mean():.2f}\n")
    
    return df_result


def create_polynomial_features(df, features, degree=2):
    """
    Generate polynomial features up to a specified degree
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    features : list
        List of feature names to create polynomials for
    degree : int
        Maximum degree of polynomial features
        
    Returns:
    --------
    DataFrame with polynomial features and feature names
    """
    print(f"\nPolynomial Features (up to degree {degree}):")
    print("=" * 70)
    print("Purpose: Capture non-linear relationships")
    print("Example: x → x², x³\n")
    
    df_result = df.copy()
    
    # Create polynomial features for each specified feature
    for feature in features:
        if feature in df.columns:
            print(f"Creating polynomials for: {feature}")
            
            for deg in range(2, degree + 1):
                new_feature_name = f'{feature}_power_{deg}'
                df_result[new_feature_name] = df[feature] ** deg
                
                print(f"  {new_feature_name}: {feature}^{deg}")
                print(f"    Range: [{df_result[new_feature_name].min():.2f}, {df_result[new_feature_name].max():.2f}]")
            print()
    
    return df_result


def create_all_polynomial_interactions(df, features, degree=2, interaction_only=False):
    """
    Use sklearn's PolynomialFeatures to create all possible interactions
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    features : list
        List of feature names
    degree : int
        Degree of polynomial features
    interaction_only : bool
        If True, only interaction features (no powers)
        
    Returns:
    --------
    DataFrame with all polynomial interaction features
    """
    print(f"\nAll Polynomial Interactions (sklearn PolynomialFeatures):")
    print("=" * 70)
    
    df_features = df[features].copy()
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df_features)
    
    # Get feature names
    feature_names = poly.get_feature_names_out(features)
    
    df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    print(f"Original features: {len(features)}")
    print(f"Total features after polynomial expansion: {len(feature_names)}")
    print(f"New interaction features created: {len(feature_names) - len(features)}\n")
    
    # Show some examples of new features
    new_features = [f for f in feature_names if f not in features]
    print(f"Examples of new features (first 10):")
    for feat in new_features[:10]:
        print(f"  - {feat}")
    
    return df_poly, feature_names


def demonstrate_interaction_importance(df):
    """
    Demonstrate how interaction features improve model performance
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with features and target
    """
    print("\n" + "=" * 70)
    print("Demonstrating the Impact of Interaction Features")
    print("=" * 70)
    
    # Prepare data - predict price based on dimensions
    base_features = ['height', 'width', 'depth', 'weight']
    target = 'price'
    
    # Model 1: Base features only
    X_base = df[base_features]
    y = df[target]
    
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )
    
    model_base = LinearRegression()
    model_base.fit(X_train_base, y_train)
    y_pred_base = model_base.predict(X_test_base)
    
    r2_base = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    
    print(f"\nModel 1: Base Features Only")
    print(f"  Features: {base_features}")
    print(f"  R² Score: {r2_base:.4f}")
    print(f"  RMSE: {rmse_base:.2f}")
    
    # Model 2: With multiplication interactions
    df_with_mult = df.copy()
    df_with_mult['volume'] = df['height'] * df['width'] * df['depth']
    df_with_mult['weight_x_volume'] = df['weight'] * df_with_mult['volume']
    
    interaction_features = base_features + ['volume', 'weight_x_volume']
    X_interaction = df_with_mult[interaction_features]
    
    X_train_int, X_test_int, _, _ = train_test_split(
        X_interaction, y, test_size=0.2, random_state=42
    )
    
    model_interaction = LinearRegression()
    model_interaction.fit(X_train_int, y_train)
    y_pred_int = model_interaction.predict(X_test_int)
    
    r2_interaction = r2_score(y_test, y_pred_int)
    rmse_interaction = np.sqrt(mean_squared_error(y_test, y_pred_int))
    
    print(f"\nModel 2: With Interaction Features")
    print(f"  Features: {interaction_features}")
    print(f"  R² Score: {r2_interaction:.4f}")
    print(f"  RMSE: {rmse_interaction:.2f}")
    
    # Model 3: With polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    
    X_train_poly, X_test_poly, _, _ = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    y_pred_poly = model_poly.predict(X_test_poly)
    
    r2_poly = r2_score(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    
    print(f"\nModel 3: With Polynomial Features (degree=2)")
    print(f"  Total features: {X_poly.shape[1]}")
    print(f"  R² Score: {r2_poly:.4f}")
    print(f"  RMSE: {rmse_poly:.2f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Performance Comparison:")
    print("=" * 70)
    
    comparison = pd.DataFrame({
        'Model': ['Base Features', 'With Interactions', 'With Polynomials'],
        'R² Score': [r2_base, r2_interaction, r2_poly],
        'RMSE': [rmse_base, rmse_interaction, rmse_poly],
        'Improvement over Base': [
            0, 
            ((r2_interaction - r2_base) / r2_base * 100),
            ((r2_poly - r2_base) / r2_base * 100)
        ]
    })
    
    print("\n", comparison.to_string(index=False))
    
    return comparison


def visualize_interactions(df):
    """Visualize the relationship between features and their interactions"""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # Create interaction features
    df_viz = df.copy()
    df_viz['volume'] = df['height'] * df['width'] * df['depth']
    df_viz['price_per_volume'] = df['price'] / (df_viz['volume'] + 1)
    
    # Visualization 1: Multiplication interaction
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Individual features vs price
    axes[0, 0].scatter(df['height'], df['price'], alpha=0.5)
    axes[0, 0].set_xlabel('Height')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].set_title('Height vs Price')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(df['width'], df['price'], alpha=0.5)
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].set_title('Width vs Price')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Interaction feature vs price
    axes[1, 0].scatter(df_viz['volume'], df['price'], alpha=0.5, color='green')
    axes[1, 0].set_xlabel('Volume (height × width × depth)')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].set_title('Volume (Interaction) vs Price - Stronger Correlation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Ratio feature
    axes[1, 1].scatter(df_viz['price_per_volume'], df['weight'], alpha=0.5, color='orange')
    axes[1, 1].set_xlabel('Price per Volume (Ratio)')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].set_title('Price per Volume vs Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task4_interactions_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task4_interactions_scatter.png")
    
    # Visualization 2: Polynomial features
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Sample one feature for polynomial demonstration
    sample_feature = df['weight'].values
    sample_target = df['price'].values
    
    for idx, degree in enumerate([1, 2, 3]):
        # Fit polynomial
        coeffs = np.polyfit(sample_feature, sample_target, degree)
        poly_func = np.poly1d(coeffs)
        
        # Plot
        axes[idx].scatter(sample_feature, sample_target, alpha=0.5)
        
        # Create smooth line
        x_smooth = np.linspace(sample_feature.min(), sample_feature.max(), 100)
        y_smooth = poly_func(x_smooth)
        axes[idx].plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Degree {degree} fit')
        
        axes[idx].set_xlabel('Weight')
        axes[idx].set_ylabel('Price')
        axes[idx].set_title(f'Polynomial Degree {degree}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task4_polynomial_fit.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task4_polynomial_fit.png")


def main():
    """Main function to demonstrate interaction features"""
    
    print("=" * 70)
    print("Task 4: Creating Interaction Features")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("PART 1: Multiplication Interactions")
    print("=" * 70)
    
    # Create multiplication interactions
    mult_pairs = [
        ('height', 'width'),
        ('height', 'depth'),
        ('width', 'depth'),
        ('height', 'width'),  # Will be used to create volume below
        ('units_sold', 'price'),
        ('store_size', 'foot_traffic')
    ]
    
    # Create volume (3-way interaction)
    df['volume'] = df['height'] * df['width'] * df['depth']
    print("\nSpecial Case - 3-way Interaction:")
    print(f"Created: volume")
    print(f"  Formula: height × width × depth")
    print(f"  Range: [{df['volume'].min():.2f}, {df['volume'].max():.2f}]")
    print(f"  Mean: {df['volume'].mean():.2f}\n")
    
    mult_pairs_2way = [
        ('height', 'width'),
        ('units_sold', 'price'),
        ('store_size', 'foot_traffic')
    ]
    
    df = create_multiplication_interactions(df, mult_pairs_2way)
    
    print("\n" + "=" * 70)
    print("PART 2: Division Interactions (Ratios)")
    print("=" * 70)
    
    # Create division interactions
    div_pairs = [
        ('price', 'volume'),
        ('revenue', 'units_sold'),
        ('units_sold', 'foot_traffic'),
        ('revenue', 'store_size'),
        ('production_time', 'volume')
    ]
    
    df = create_division_interactions(df, div_pairs)
    
    print("\n" + "=" * 70)
    print("PART 3: Polynomial Features")
    print("=" * 70)
    
    # Create polynomial features
    poly_features = ['weight', 'foot_traffic', 'store_size']
    df = create_polynomial_features(df, poly_features, degree=3)
    
    print("\n" + "=" * 70)
    print("PART 4: All Polynomial Interactions (sklearn)")
    print("=" * 70)
    
    # Create all polynomial interactions using sklearn
    selected_features = ['height', 'width', 'depth', 'weight']
    df_poly, poly_feature_names = create_all_polynomial_interactions(
        df, selected_features, degree=2, interaction_only=False
    )
    
    print("\n" + "=" * 70)
    print("PART 5: Impact on Model Performance")
    print("=" * 70)
    
    # Demonstrate model improvement
    comparison_df = demonstrate_interaction_importance(df)
    
    print("\n" + "=" * 70)
    print("PART 6: Visualizations")
    print("=" * 70)
    
    # Create visualizations
    visualize_interactions(df)
    
    # Show summary of all created features
    print("\n" + "=" * 70)
    print("Summary of Created Features")
    print("=" * 70)
    
    original_features = ['height', 'width', 'depth', 'weight', 'price', 
                         'production_time', 'units_sold', 'revenue', 
                         'store_size', 'foot_traffic']
    
    new_features = [col for col in df.columns if col not in original_features]
    
    print(f"\nOriginal features: {len(original_features)}")
    print(f"New features created: {len(new_features)}")
    print(f"Total features: {len(df.columns)}")
    
    print(f"\nNew feature types:")
    mult_features = [f for f in new_features if '_x_' in f or f == 'volume']
    div_features = [f for f in new_features if '_per_' in f]
    poly_features = [f for f in new_features if '_power_' in f]
    
    print(f"  Multiplication interactions: {len(mult_features)}")
    print(f"  Division interactions (ratios): {len(div_features)}")
    print(f"  Polynomial features: {len(poly_features)}")
    
    # Save the dataset with all features
    df.to_csv('task4_with_interactions.csv', index=False)
    print(f"\nSaved dataset with all interaction features to: task4_with_interactions.csv")
    
    print("\n" + "=" * 70)
    print("Task 4 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Multiplication interactions capture combined effects (e.g., area = height × width)")
    print("2. Division interactions create meaningful ratios and rates")
    print("3. Polynomial features capture non-linear relationships")
    print("4. Interaction features significantly improve linear model performance")
    print("5. Use domain knowledge to create meaningful interactions")
    print("6. sklearn's PolynomialFeatures automates interaction creation")
    print("7. Balance between model complexity and interpretability")
    print("8. Feature engineering often provides better gains than complex models")


if __name__ == "__main__":
    main()
