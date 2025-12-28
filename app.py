"""
Streamlit Dashboard for MLDS Sprint 10 Examples
Visualizes results from all 7 data preprocessing and feature engineering tasks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="MLDS Sprint 10 - Data Preprocessing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .task-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_image(image_path):
    """Load and display an image if it exists"""
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None


def display_task1():
    """Task 1: Outlier Detection and Treatment"""
    st.markdown('<p class="task-header">Task 1: Outlier Detection and Treatment</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    This task demonstrates how to detect and handle outliers using the IQR (Interquartile Range) method.
    
    **Key Concepts:**
    - **IQR Method**: Q1 - 1.5×IQR (lower bound) and Q3 + 1.5×IQR (upper bound)
    - **Capping**: Replace outliers with boundary values
    - **Removal**: Remove rows containing outliers
    """)
    
    # Display visualizations
    st.subheader("📊 Visualizations")
    
    cols = st.columns(3)
    for idx, feature in enumerate(['age', 'salary', 'experience']):
        img_path = f'task1_outliers_{feature}.png'
        img = load_image(img_path)
        if img:
            with cols[idx]:
                st.image(img, caption=f'{feature.capitalize()} Outlier Treatment', use_column_width=True)
    
    # Key takeaways
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **IQR method** effectively identifies outliers using statistical boundaries
    2. **Capping** preserves all data points while limiting extreme values
    3. **Removal** eliminates outliers but reduces dataset size
    4. Always **visualize** data before and after treatment
    5. Choose method based on data context and analysis goals
    """)


def display_task2():
    """Task 2: Feature Binning and Discretization"""
    st.markdown('<p class="task-header">Task 2: Feature Binning and Discretization</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Feature binning converts continuous variables into categorical bins, improving model interpretability.
    
    **Binning Methods:**
    - **Equal-Width**: Uniform intervals (may have unbalanced counts)
    - **Equal-Frequency**: Balanced bins (intervals may vary)
    - **Custom**: Domain-specific categories (e.g., age groups, income brackets)
    """)
    
    # Display visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        img = load_image('task2_binning_age.png')
        if img:
            st.image(img, caption='Age Binning Comparison', use_column_width=True)
    
    with col2:
        img = load_image('task2_binning_income.png')
        if img:
            st.image(img, caption='Income Binning Comparison', use_column_width=True)
    
    # Load and display data
    if os.path.exists('task2_binned_data.csv'):
        st.subheader("📋 Sample Data")
        df = pd.read_csv('task2_binned_data.csv')
        st.dataframe(df.head(10))
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **Equal-width** creates uniform intervals but may result in unbalanced counts
    2. **Equal-frequency** ensures balanced bins but intervals vary
    3. **Custom binning** leverages domain knowledge for meaningful categories
    4. Binning improves model **interpretability** and handles **non-linear** relationships
    """)


def display_task3():
    """Task 3: Advanced Data Imputation"""
    st.markdown('<p class="task-header">Task 3: Advanced Data Imputation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Advanced imputation techniques go beyond simple mean/median/mode imputation.
    
    **Imputation Methods:**
    - **KNN Imputation**: Uses k-nearest neighbors to fill missing values
    - **Iterative (MICE)**: Models each feature as function of others
    - **Forward/Backward Fill**: Propagates values temporally
    - **Constant Value**: Domain-specific defaults
    """)
    
    # Display visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        img = load_image('task3_imputation_age.png')
        if img:
            st.image(img, caption='Age Imputation Comparison', use_column_width=True)
    
    with col2:
        img = load_image('task3_imputation_income.png')
        if img:
            st.image(img, caption='Income Imputation Comparison', use_column_width=True)
    
    # Load imputed data comparison
    st.subheader("📊 Imputation Methods Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("KNN Imputation", "Best for", "Local patterns")
        st.info("Considers feature relationships and similar observations")
    
    with col2:
        st.metric("MICE (Iterative)", "Best for", "Complex relationships")
        st.info("Captures multivariate patterns through iteration")
    
    with col3:
        st.metric("Forward/Backward Fill", "Best for", "Time-series")
        st.info("Maintains temporal consistency")
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **KNN** leverages feature similarity for accurate imputation
    2. **MICE** captures complex multivariate relationships
    3. **Forward/backward fill** preserves temporal patterns
    4. Method choice depends on **data characteristics** and **missing patterns**
    """)


def display_task4():
    """Task 4: Creating Interaction Features"""
    st.markdown('<p class="task-header">Task 4: Creating Interaction Features</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Interaction features capture relationships between existing features.
    
    **Interaction Types:**
    - **Multiplication**: height × width = area
    - **Division**: price / volume = price_per_unit
    - **Polynomial**: x → x², x³
    """)
    
    # Display visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        img = load_image('task4_interactions_scatter.png')
        if img:
            st.image(img, caption='Interaction Features vs Individual Features', use_column_width=True)
    
    with col2:
        img = load_image('task4_polynomial_fit.png')
        if img:
            st.image(img, caption='Polynomial Features of Different Degrees', use_column_width=True)
    
    # Model performance comparison
    st.subheader("🎯 Model Performance Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Base Features", "R² = 0.975", "Baseline")
        st.caption("4 features: height, width, depth, weight")
    
    with col2:
        st.metric("With Interactions", "R² = 0.9996", "+2.5%")
        st.caption("Added volume and weight×volume")
    
    with col3:
        st.metric("With Polynomials", "R² = 0.9995", "+2.5%")
        st.caption("14 features including all interactions")
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **Multiplication** interactions capture combined effects
    2. **Division** interactions create meaningful ratios
    3. **Polynomial** features capture non-linear relationships
    4. Interaction features **significantly improve** linear model performance
    5. Use **domain knowledge** to create meaningful interactions
    """)


def display_task5():
    """Task 5: Ordinal Encoding"""
    st.markdown('<p class="task-header">Task 5: Multi-Class Label Encoding with Ordinal Relationships</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Ordinal encoding preserves natural order of categorical variables.
    
    **Examples:**
    - Education: High School (0) → Bachelor's (1) → Master's (2) → PhD (3)
    - Experience: Entry (0) → Junior (1) → Mid (2) → Senior (3) → Expert (4)
    - Ratings: Poor (0) → Fair (1) → Good (2) → Very Good (3) → Excellent (4)
    """)
    
    # Display visualization
    st.subheader("📊 Visualizations")
    
    img = load_image('task5_ordinal_encoding.png')
    if img:
        st.image(img, caption='Ordinal Encoding Analysis', use_column_width=True)
    
    # Comparison
    st.subheader("🔄 Ordinal vs One-Hot Encoding")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Ordinal Encoding")
        st.success("✓ 5 features")
        st.success("✓ 83% accuracy")
        st.success("✓ Preserves order")
        st.success("✓ Less complex")
    
    with col2:
        st.markdown("### One-Hot Encoding")
        st.warning("16 features (+220%)")
        st.warning("81% accuracy")
        st.warning("No order information")
        st.warning("More complex")
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **Preserves natural order** (Low < Medium < High)
    2. **68.8% feature reduction** compared to one-hot encoding
    3. **Better performance** with tree-based models
    4. Handle **unknown categories** with default value (-1)
    5. Always **validate** that encoding preserves intended order
    """)


def display_task6():
    """Task 6: Time-Based Feature Engineering"""
    st.markdown('<p class="task-header">Task 6: Time-Based Feature Engineering</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Extract powerful features from datetime information.
    
    **Feature Categories:**
    - **Date Components**: year, month, day, hour, quarter, day of week
    - **Cyclic Features**: sin/cos transformations for hour, month, day of week
    - **Time Differences**: days since event, age calculations
    - **Business Features**: is_weekend, is_business_hours, is_holiday_season
    """)
    
    # Display visualization
    st.subheader("📊 Visualizations")
    
    img = load_image('task6_temporal_features.png')
    if img:
        st.image(img, caption='Temporal Feature Analysis', use_column_width=True)
    
    # Key statistics
    st.subheader("📈 Temporal Patterns Discovered")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Weekend Transactions", "$164.82 avg", "+51% higher")
        st.caption("vs $109.18 on weekdays")
    
    with col2:
        st.metric("Business Hours", "$142.77 avg", "+23% higher")
        st.caption("vs $116.43 non-business hours")
    
    with col3:
        st.metric("Features Created", "36 features", "from 1 datetime")
        st.caption("Extracted from transaction_datetime")
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. Extract **granular components** (year, month, day, hour)
    2. **Cyclic encoding** (sin/cos) preserves periodic nature
    3. **Time differences** capture temporal relationships
    4. **Business features** reveal behavioral patterns
    5. Temporal features **significantly improve** time-series predictions
    """)


def display_task7():
    """Task 7: Handling Imbalanced Data"""
    st.markdown('<p class="task-header">Task 7: Handling Imbalanced Data with Resampling</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Overview
    Resampling techniques address class imbalance in classification problems.
    
    **Original Dataset:**
    - Class 0 (Majority): 89.6%
    - Class 1 (Minority): 10.4%
    - Imbalance Ratio: 8.6:1
    """)
    
    # Display visualizations
    st.subheader("📊 Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        img = load_image('task7_distributions.png')
        if img:
            st.image(img, caption='Class Distribution After Resampling', use_column_width=True)
    
    with col2:
        img = load_image('task7_confusion_matrices.png')
        if img:
            st.image(img, caption='Confusion Matrices Comparison', use_column_width=True)
    
    # Performance comparison
    st.subheader("🏆 Performance Comparison")
    
    if os.path.exists('task7_resampling_results.csv'):
        df_results = pd.read_csv('task7_resampling_results.csv')
        
        # Display metrics table
        st.dataframe(
            df_results[['Method', 'Accuracy', 'F1-Score', 'ROC-AUC', 'Sensitivity', 'Specificity']].round(3)
        )
        
        # Best method highlight
        best_idx = df_results['F1-Score'].idxmax()
        best_method = df_results.loc[best_idx, 'Method']
        best_f1 = float(df_results.loc[best_idx, 'F1-Score'])
        baseline_f1 = float(df_results.loc[0, 'F1-Score'])
        improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
        
        st.success(f"🏆 **Best Method: {best_method}** - F1-Score: {best_f1:.3f} (+{improvement:.1f}% improvement)")
    
    # Method comparison
    st.subheader("🔍 Method Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Random Oversampling")
        st.info("✓ Simple and fast\n\n✓ Preserves all data")
        st.warning("✗ May overfit\n\n✗ Increases size")
    
    with col2:
        st.markdown("### SMOTE")
        st.info("✓ Synthetic samples\n\n✓ Better performance")
        st.warning("✗ May create noise\n\n✗ More complex")
    
    with col3:
        st.markdown("### SMOTE + Cleaning")
        st.info("✓ Cleans boundaries\n\n✓ Best quality")
        st.warning("✗ Most complex\n\n✗ May remove data")
    
    # Metrics visualization
    img = load_image('task7_metrics_comparison.png')
    if img:
        st.image(img, caption='Detailed Metrics Comparison', use_column_width=True)
    
    st.markdown("""
    ### 🎯 Key Takeaways
    1. **Imbalanced data** biases models toward majority class
    2. **Random oversampling** duplicates minority samples
    3. **Random undersampling** removes majority samples
    4. **SMOTE** generates synthetic samples via interpolation
    5. **Combination methods** (SMOTE + cleaning) often perform best
    6. Use **F1-score and ROC-AUC** for imbalanced data evaluation
    """)


def display_overview():
    """Display overview page"""
    st.markdown('<p class="main-header">MLDS Sprint 10: Data Preprocessing & Feature Engineering</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📚 Welcome to the Interactive Dashboard
    
    This dashboard presents comprehensive solutions to 7 critical data preprocessing and feature engineering tasks.
    Each task includes working code, visualizations, and detailed analysis.
    
    ---
    """)
    
    # Task summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Tasks Overview")
        st.markdown("""
        1. **Outlier Detection & Treatment** - IQR method with capping and removal
        2. **Feature Binning** - Equal-width, equal-frequency, and custom binning
        3. **Advanced Imputation** - KNN, MICE, forward/backward fill
        4. **Interaction Features** - Multiplication, division, polynomial features
        """)
    
    with col2:
        st.markdown("### 🎯 Tasks Overview (cont.)")
        st.markdown("""
        5. **Ordinal Encoding** - Preserve natural order of categories
        6. **Temporal Features** - Extract 36+ features from datetime
        7. **Imbalanced Data** - SMOTE and resampling techniques
        """)
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Dashboard Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tasks", "7", "Complete")
    
    with col2:
        st.metric("Visualizations", "15+", "PNG files")
    
    with col3:
        st.metric("Datasets", "10+", "CSV files")
    
    with col4:
        st.metric("Python Scripts", "7", "Executable")
    
    st.markdown("---")
    
    # Getting started
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Select a task** from the sidebar to explore detailed results
    2. **View visualizations** showing before/after comparisons
    3. **Read key takeaways** and implementation notes
    4. **Download sample data** to try the techniques yourself
    
    ### 💻 Running the Examples
    
    Each task has a corresponding Python script:
    ```bash
    python task1_outlier_detection.py
    python task2_feature_binning.py
    python task3_advanced_imputation.py
    python task4_interaction_features.py
    python task5_ordinal_encoding.py
    python task6_temporal_features.py
    python task7_imbalanced_data.py
    ```
    
    ### 📦 Requirements
    
    Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    
    ### 📖 Resources
    
    - All code is well-documented with inline comments
    - Each script generates visualizations and CSV files
    - Results are reproducible with fixed random seeds
    """)
    
    st.markdown("---")
    st.info("👈 **Select a task from the sidebar to begin exploring!**")


def main():
    """Main Streamlit app"""
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    task_options = {
        "🏠 Overview": "overview",
        "Task 1: Outlier Detection": "task1",
        "Task 2: Feature Binning": "task2",
        "Task 3: Advanced Imputation": "task3",
        "Task 4: Interaction Features": "task4",
        "Task 5: Ordinal Encoding": "task5",
        "Task 6: Temporal Features": "task6",
        "Task 7: Imbalanced Data": "task7"
    }
    
    selected_task = st.sidebar.radio("Select Task:", list(task_options.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📌 About
    
    This dashboard showcases 7 essential data preprocessing techniques for machine learning.
    
    **Author:** MLDS Sprint 10  
    **Date:** December 2025
    
    ---
    
    ### 🔗 Quick Links
    - [View Code](.)
    - [Download Data](.)
    - [Documentation](README.md)
    """)
    
    # Display selected task
    task = task_options[selected_task]
    
    if task == "overview":
        display_overview()
    elif task == "task1":
        display_task1()
    elif task == "task2":
        display_task2()
    elif task == "task3":
        display_task3()
    elif task == "task4":
        display_task4()
    elif task == "task5":
        display_task5()
    elif task == "task6":
        display_task6()
    elif task == "task7":
        display_task7()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>MLDS Sprint 10 - Data Preprocessing & Feature Engineering Dashboard</p>
        <p>Built with Streamlit | Python | Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
