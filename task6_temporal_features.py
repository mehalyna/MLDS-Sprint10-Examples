"""
Task 6: Time-Based Feature Engineering
Demonstrates extracting date components, creating cyclic features, and calculating time differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data():
    """Create a sample dataset with datetime information"""
    n_samples = 1000
    
    # Generate random dates over 2 years
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
    
    # Transaction timestamps
    transaction_times = [
        datetime.combine(d.date(), datetime.min.time()) + 
        timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
        for d in dates
    ]
    
    # Account creation dates (earlier than transactions)
    account_creation = [
        d - timedelta(days=np.random.randint(30, 1095))  # 1 month to 3 years before
        for d in dates
    ]
    
    # Birth dates (for age calculation)
    birth_dates = [
        datetime(np.random.randint(1960, 2000), np.random.randint(1, 13), np.random.randint(1, 28))
        for _ in range(n_samples)
    ]
    
    # Generate transaction amounts (influenced by time)
    amounts = []
    for dt in transaction_times:
        base_amount = 100
        # Higher amounts on weekends
        if dt.weekday() >= 5:
            base_amount *= 1.5
        # Higher amounts during holiday season
        if dt.month == 12:
            base_amount *= 1.3
        # Higher amounts during business hours
        if 9 <= dt.hour <= 17:
            base_amount *= 1.2
        
        amount = base_amount + np.random.normal(0, 30)
        amounts.append(max(10, amount))
    
    df = pd.DataFrame({
        'transaction_datetime': transaction_times,
        'account_created_date': account_creation,
        'birth_date': birth_dates,
        'amount': amounts,
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], n_samples)
    })
    
    return df


def extract_date_components(df, datetime_column):
    """
    Extract various date and time components from a datetime column
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    datetime_column : str
        Name of the datetime column
        
    Returns:
    --------
    DataFrame with extracted date components
    """
    print(f"\nExtracting Date Components from '{datetime_column}':")
    print("=" * 70)
    
    df_result = df.copy()
    dt_col = df_result[datetime_column]
    
    # Basic components
    df_result[f'{datetime_column}_year'] = dt_col.dt.year
    df_result[f'{datetime_column}_month'] = dt_col.dt.month
    df_result[f'{datetime_column}_day'] = dt_col.dt.day
    df_result[f'{datetime_column}_dayofweek'] = dt_col.dt.dayofweek  # Monday=0, Sunday=6
    df_result[f'{datetime_column}_dayofyear'] = dt_col.dt.dayofyear
    df_result[f'{datetime_column}_hour'] = dt_col.dt.hour
    df_result[f'{datetime_column}_minute'] = dt_col.dt.minute
    df_result[f'{datetime_column}_quarter'] = dt_col.dt.quarter
    
    # Binary flags
    df_result[f'{datetime_column}_is_weekend'] = (dt_col.dt.dayofweek >= 5).astype(int)
    df_result[f'{datetime_column}_is_month_start'] = dt_col.dt.is_month_start.astype(int)
    df_result[f'{datetime_column}_is_month_end'] = dt_col.dt.is_month_end.astype(int)
    df_result[f'{datetime_column}_is_quarter_start'] = dt_col.dt.is_quarter_start.astype(int)
    df_result[f'{datetime_column}_is_quarter_end'] = dt_col.dt.is_quarter_end.astype(int)
    
    # Day name and month name
    df_result[f'{datetime_column}_day_name'] = dt_col.dt.day_name()
    df_result[f'{datetime_column}_month_name'] = dt_col.dt.month_name()
    
    # Week of year
    df_result[f'{datetime_column}_week_of_year'] = dt_col.dt.isocalendar().week
    
    print("\nExtracted Components:")
    components = [
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'hour', 'minute', 'quarter',
        'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'day_name', 'month_name', 'week_of_year'
    ]
    
    for comp in components:
        col_name = f'{datetime_column}_{comp}'
        if col_name in df_result.columns:
            if df_result[col_name].dtype in ['int64', 'int32']:
                print(f"  {col_name:45s}: Range [{df_result[col_name].min()}, {df_result[col_name].max()}]")
            else:
                print(f"  {col_name:45s}: {df_result[col_name].nunique()} unique values")
    
    return df_result


def create_cyclic_features(df, column, max_value):
    """
    Create sine and cosine transformations for cyclic features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column : str
        Column name to transform
    max_value : int
        Maximum value for normalization (e.g., 24 for hours, 12 for months)
        
    Returns:
    --------
    DataFrame with cyclic features
    """
    df_result = df.copy()
    
    # Normalize to [0, 1] range
    normalized = df_result[column] / max_value
    
    # Create sine and cosine features
    df_result[f'{column}_sin'] = np.sin(2 * np.pi * normalized)
    df_result[f'{column}_cos'] = np.cos(2 * np.pi * normalized)
    
    print(f"\nCreated cyclic features for '{column}' (max={max_value}):")
    print(f"  {column}_sin: Range [{df_result[f'{column}_sin'].min():.3f}, {df_result[f'{column}_sin'].max():.3f}]")
    print(f"  {column}_cos: Range [{df_result[f'{column}_cos'].min():.3f}, {df_result[f'{column}_cos'].max():.3f}]")
    
    return df_result


def calculate_time_differences(df, datetime_column, reference_date=None):
    """
    Calculate time differences from a reference date
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    datetime_column : str
        Column name containing datetime
    reference_date : datetime or str
        Reference date (if None, uses current date)
        
    Returns:
    --------
    DataFrame with time difference features
    """
    print(f"\nCalculating Time Differences for '{datetime_column}':")
    print("=" * 70)
    
    df_result = df.copy()
    
    if reference_date is None:
        reference_date = datetime.now()
    elif isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)
    
    dt_col = pd.to_datetime(df_result[datetime_column])
    
    # Calculate differences
    time_delta = reference_date - dt_col
    
    df_result[f'{datetime_column}_days_since'] = time_delta.dt.days
    df_result[f'{datetime_column}_weeks_since'] = (time_delta.dt.days / 7).astype(int)
    df_result[f'{datetime_column}_months_since'] = (time_delta.dt.days / 30.44).astype(int)  # Average month length
    df_result[f'{datetime_column}_years_since'] = (time_delta.dt.days / 365.25).astype(int)  # Accounting for leap years
    
    print(f"Reference date: {reference_date.strftime('%Y-%m-%d')}")
    print(f"\nTime difference features:")
    print(f"  days_since   : Range [{df_result[f'{datetime_column}_days_since'].min()}, {df_result[f'{datetime_column}_days_since'].max()}]")
    print(f"  weeks_since  : Range [{df_result[f'{datetime_column}_weeks_since'].min()}, {df_result[f'{datetime_column}_weeks_since'].max()}]")
    print(f"  months_since : Range [{df_result[f'{datetime_column}_months_since'].min()}, {df_result[f'{datetime_column}_months_since'].max()}]")
    print(f"  years_since  : Range [{df_result[f'{datetime_column}_years_since'].min()}, {df_result[f'{datetime_column}_years_since'].max()}]")
    
    return df_result


def calculate_age(df, birth_date_column, reference_date=None):
    """
    Calculate age from birth date
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    birth_date_column : str
        Column containing birth dates
    reference_date : datetime
        Reference date for age calculation (default: current date)
        
    Returns:
    --------
    DataFrame with age feature
    """
    print(f"\nCalculating Age from '{birth_date_column}':")
    print("=" * 70)
    
    df_result = df.copy()
    
    if reference_date is None:
        reference_date = datetime.now()
    
    birth_dates = pd.to_datetime(df_result[birth_date_column])
    
    # Calculate age in years
    df_result['age_years'] = ((reference_date - birth_dates).dt.days / 365.25).astype(int)
    
    # Create age groups
    df_result['age_group'] = pd.cut(
        df_result['age_years'],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )
    
    print(f"Age statistics:")
    print(f"  Min age: {df_result['age_years'].min()} years")
    print(f"  Max age: {df_result['age_years'].max()} years")
    print(f"  Mean age: {df_result['age_years'].mean():.1f} years")
    print(f"\nAge group distribution:")
    print(df_result['age_group'].value_counts().sort_index())
    
    return df_result


def calculate_time_between_events(df, datetime_col1, datetime_col2, feature_name):
    """
    Calculate time difference between two datetime columns
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    datetime_col1 : str
        First datetime column (later date)
    datetime_col2 : str
        Second datetime column (earlier date)
    feature_name : str
        Name for the new feature
        
    Returns:
    --------
    DataFrame with time difference feature
    """
    print(f"\nCalculating Time Between Events:")
    print(f"  {datetime_col1} - {datetime_col2}")
    print("=" * 70)
    
    df_result = df.copy()
    
    dt1 = pd.to_datetime(df_result[datetime_col1])
    dt2 = pd.to_datetime(df_result[datetime_col2])
    
    time_diff = (dt1 - dt2).dt.days
    
    df_result[f'{feature_name}_days'] = time_diff
    df_result[f'{feature_name}_weeks'] = (time_diff / 7).astype(int)
    df_result[f'{feature_name}_months'] = (time_diff / 30.44).astype(int)
    
    print(f"\n{feature_name} statistics:")
    print(f"  Mean: {time_diff.mean():.1f} days")
    print(f"  Median: {time_diff.median():.1f} days")
    print(f"  Min: {time_diff.min()} days")
    print(f"  Max: {time_diff.max()} days")
    
    return df_result


def create_business_features(df, datetime_column):
    """
    Create business-related temporal features
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    datetime_column : str
        Datetime column name
        
    Returns:
    --------
    DataFrame with business features
    """
    print(f"\nCreating Business Features from '{datetime_column}':")
    print("=" * 70)
    
    df_result = df.copy()
    dt_col = pd.to_datetime(df_result[datetime_column])
    
    # Business hours
    df_result['is_business_hours'] = ((dt_col.dt.hour >= 9) & (dt_col.dt.hour <= 17)).astype(int)
    
    # Time of day categories
    hour = dt_col.dt.hour
    df_result['time_of_day'] = pd.cut(
        hour,
        bins=[-1, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    
    # Season
    month = dt_col.dt.month
    df_result['season'] = pd.cut(
        month,
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall']
    )
    
    # Holiday season
    df_result['is_holiday_season'] = dt_col.dt.month.isin([11, 12]).astype(int)
    
    # Business day (Monday-Friday)
    df_result['is_business_day'] = (dt_col.dt.dayofweek < 5).astype(int)
    
    print("\nCreated business features:")
    print(f"  is_business_hours  : {df_result['is_business_hours'].sum()} transactions ({df_result['is_business_hours'].mean()*100:.1f}%)")
    print(f"  is_business_day    : {df_result['is_business_day'].sum()} transactions ({df_result['is_business_day'].mean()*100:.1f}%)")
    print(f"  is_holiday_season  : {df_result['is_holiday_season'].sum()} transactions ({df_result['is_holiday_season'].mean()*100:.1f}%)")
    print(f"\n  Time of day distribution:")
    print(df_result['time_of_day'].value_counts().sort_index())
    print(f"\n  Season distribution:")
    print(df_result['season'].value_counts().sort_index())
    
    return df_result


def visualize_temporal_features(df):
    """Visualize temporal features and patterns"""
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # 1. Transactions by hour of day
    hourly_counts = df.groupby('transaction_datetime_hour').size()
    axes[0, 0].bar(hourly_counts.index, hourly_counts.values, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Number of Transactions')
    axes[0, 0].set_title('Transaction Distribution by Hour')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Transactions by day of week
    dow_counts = df.groupby('transaction_datetime_day_name').size().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    axes[0, 1].bar(range(len(dow_counts)), dow_counts.values, color='coral', edgecolor='black')
    axes[0, 1].set_xticks(range(len(dow_counts)))
    axes[0, 1].set_xticklabels(dow_counts.index, rotation=45, ha='right')
    axes[0, 1].set_xlabel('Day of Week')
    axes[0, 1].set_ylabel('Number of Transactions')
    axes[0, 1].set_title('Transaction Distribution by Day of Week')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Cyclic feature visualization (hour)
    scatter = axes[1, 0].scatter(
        df['transaction_datetime_hour_cos'],
        df['transaction_datetime_hour_sin'],
        c=df['transaction_datetime_hour'],
        cmap='twilight',
        alpha=0.6
    )
    axes[1, 0].set_xlabel('Hour (Cosine)')
    axes[1, 0].set_ylabel('Hour (Sine)')
    axes[1, 0].set_title('Cyclic Encoding of Hour (24-hour cycle)')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Hour')
    
    # 4. Average amount by time of day
    time_of_day_avg = df.groupby('time_of_day')['amount'].mean().reindex(
        ['Night', 'Morning', 'Afternoon', 'Evening']
    )
    axes[1, 1].bar(range(len(time_of_day_avg)), time_of_day_avg.values, color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(range(len(time_of_day_avg)))
    axes[1, 1].set_xticklabels(time_of_day_avg.index)
    axes[1, 1].set_xlabel('Time of Day')
    axes[1, 1].set_ylabel('Average Transaction Amount ($)')
    axes[1, 1].set_title('Average Transaction Amount by Time of Day')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 5. Transactions by month
    monthly_counts = df.groupby('transaction_datetime_month').size()
    axes[2, 0].plot(monthly_counts.index, monthly_counts.values, marker='o', linewidth=2, markersize=8, color='purple')
    axes[2, 0].set_xlabel('Month')
    axes[2, 0].set_ylabel('Number of Transactions')
    axes[2, 0].set_title('Transaction Distribution by Month')
    axes[2, 0].set_xticks(range(1, 13))
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Weekend vs Weekday comparison
    weekend_comparison = df.groupby('transaction_datetime_is_weekend')['amount'].mean()
    labels = ['Weekday', 'Weekend']
    axes[2, 1].bar(range(len(weekend_comparison)), weekend_comparison.values, 
                   color=['skyblue', 'orange'], edgecolor='black')
    axes[2, 1].set_xticks(range(len(weekend_comparison)))
    axes[2, 1].set_xticklabels(labels)
    axes[2, 1].set_ylabel('Average Transaction Amount ($)')
    axes[2, 1].set_title('Average Transaction Amount: Weekday vs Weekend')
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('task6_temporal_features.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: task6_temporal_features.png")


def main():
    """Main function to demonstrate time-based feature engineering"""
    
    print("=" * 70)
    print("Task 6: Time-Based Feature Engineering")
    print("=" * 70)
    
    # Create sample data
    df = create_sample_data()
    print(f"\nOriginal dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(10))
    
    print("\n" + "=" * 70)
    print("PART 1: Extract Date Components")
    print("=" * 70)
    
    # Extract date components from transaction datetime
    df = extract_date_components(df, 'transaction_datetime')
    
    print("\n" + "=" * 70)
    print("PART 2: Create Cyclic Features")
    print("=" * 70)
    
    print("\nCyclic features capture periodic nature of time:")
    print("  - Hour of day repeats every 24 hours")
    print("  - Month repeats every 12 months")
    print("  - Day of week repeats every 7 days")
    print("\nFormula: sin(2π × value / max) and cos(2π × value / max)")
    
    # Create cyclic features for hour
    df = create_cyclic_features(df, 'transaction_datetime_hour', 24)
    
    # Create cyclic features for month
    df = create_cyclic_features(df, 'transaction_datetime_month', 12)
    
    # Create cyclic features for day of week
    df = create_cyclic_features(df, 'transaction_datetime_dayofweek', 7)
    
    print("\n" + "=" * 70)
    print("PART 3: Calculate Time Differences")
    print("=" * 70)
    
    # Calculate days since account creation
    df = calculate_time_between_events(
        df, 
        'transaction_datetime', 
        'account_created_date',
        'account_age'
    )
    
    # Calculate age from birth date
    df = calculate_age(df, 'birth_date', reference_date=datetime(2024, 12, 31))
    
    # Calculate time since transaction (from a reference date)
    df = calculate_time_differences(df, 'transaction_datetime', reference_date='2024-12-31')
    
    print("\n" + "=" * 70)
    print("PART 4: Create Business Features")
    print("=" * 70)
    
    df = create_business_features(df, 'transaction_datetime')
    
    print("\n" + "=" * 70)
    print("PART 5: Feature Summary")
    print("=" * 70)
    
    # Count all new features
    original_features = ['transaction_datetime', 'account_created_date', 'birth_date', 'amount', 'category']
    new_features = [col for col in df.columns if col not in original_features]
    
    print(f"\nOriginal features: {len(original_features)}")
    print(f"New temporal features: {len(new_features)}")
    print(f"Total features: {len(df.columns)}")
    
    # Categorize features
    date_component_features = [f for f in new_features if any(x in f for x in ['_year', '_month', '_day', '_hour', '_minute', '_quarter', '_week'])]
    cyclic_features = [f for f in new_features if '_sin' in f or '_cos' in f]
    time_diff_features = [f for f in new_features if 'since' in f or 'age' in f]
    business_features = [f for f in new_features if 'is_' in f or 'season' in f or 'time_of_day' in f]
    
    print(f"\nFeature categories:")
    print(f"  Date components: {len(date_component_features)}")
    print(f"  Cyclic features: {len(cyclic_features)}")
    print(f"  Time differences: {len(time_diff_features)}")
    print(f"  Business features: {len(business_features)}")
    
    print("\n" + "=" * 70)
    print("PART 6: Analyze Temporal Patterns")
    print("=" * 70)
    
    # Weekend vs weekday analysis
    print("\nWeekend vs Weekday Analysis:")
    weekend_stats = df.groupby('transaction_datetime_is_weekend')['amount'].agg(['mean', 'count', 'sum'])
    weekend_stats.index = ['Weekday', 'Weekend']
    print(weekend_stats)
    
    # Business hours analysis
    print("\nBusiness Hours vs Non-Business Hours:")
    business_hours_stats = df.groupby('is_business_hours')['amount'].agg(['mean', 'count', 'sum'])
    business_hours_stats.index = ['Non-Business', 'Business Hours']
    print(business_hours_stats)
    
    print("\n" + "=" * 70)
    print("PART 7: Visualizations")
    print("=" * 70)
    
    # Create visualizations
    visualize_temporal_features(df)
    
    # Save dataset with all temporal features
    df.to_csv('task6_temporal_features.csv', index=False)
    print(f"\nSaved dataset to: task6_temporal_features.csv")
    
    print("\n" + "=" * 70)
    print("Task 6 Completed Successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Extract granular date components (year, month, day, hour, etc.)")
    print("2. Cyclic encoding preserves periodic nature of time (sine/cosine)")
    print("3. Time differences capture temporal relationships between events")
    print("4. Age calculations provide demographic insights")
    print("5. Business features (weekday, business hours) capture domain patterns")
    print("6. Temporal features reveal seasonality and trends")
    print("7. Weekend/holiday flags capture behavioral differences")
    print("8. Time-based features significantly improve time-series predictions")


if __name__ == "__main__":
    main()
