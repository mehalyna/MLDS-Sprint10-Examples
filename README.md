# MLDS-Sprint10-Examples

## Data Preprocessing and Feature Engineering Tasks

## Setup Instructions

### 1. Create and Activate Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Individual Tasks
Execute any task script:
```bash
python task1_outlier_detection.py
python task2_feature_binning.py
python task3_advanced_imputation.py
python task4_interaction_features.py
python task5_ordinal_encoding.py
python task6_temporal_features.py
python task7_imbalanced_data.py
```

### 4. Launch Streamlit Dashboard
To view all task results interactively:
```bash
streamlit run app.py
```
The dashboard will open in your browser at `http://localhost:8501`

---

### Task 1: Outlier Detection and Treatment
You are provided with a dataset that contains numerical features with potential outliers. Your task is to detect and handle outliers using multiple approaches:

**Detect Outliers Using IQR Method**: Identify outliers by calculating the Interquartile Range (IQR) for each numerical column. Values that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR should be flagged as outliers.

**Handle Outliers**: After detection, apply two different treatment methods:
- **Capping**: Cap the outliers by replacing them with the upper or lower boundary values (Q1 - 1.5 * IQR and Q3 + 1.5 * IQR).
- **Removal**: Remove rows containing outliers if they represent less than 5% of the total dataset.

This task is essential for ensuring that extreme values don't negatively impact your machine learning models.

### Task 2: Feature Binning and Discretization
You are provided with a dataset containing continuous numerical features. Your task is to convert these continuous variables into categorical bins, which can help improve model performance and interpretability:

**Equal-Width Binning**: Divide the range of a continuous feature (e.g., age) into equal-width bins (e.g., 0-20, 21-40, 41-60, 61-80, 81-100).

**Equal-Frequency Binning**: Create bins such that each bin contains approximately the same number of observations. This is useful when the data distribution is skewed.

**Custom Binning**: Create bins based on domain knowledge. For example, for income data, you might create bins like "Low Income" (< $30k), "Middle Income" ($30k-$75k), "Upper Middle Income" ($75k-$150k), and "High Income" (> $150k).

This task teaches you how to transform continuous data into categorical features for better model interpretability and performance.

### Task 3: Advanced Data Imputation with Multiple Techniques
You are provided with a dataset containing missing values across multiple columns with different data types. Your task is to apply advanced imputation techniques beyond simple mean/median/mode imputation:

**K-Nearest Neighbors (KNN) Imputation**: Impute missing values by finding the k-nearest neighbors of each sample with missing values and using their average (for numerical features) or most common value (for categorical features) to fill in the gaps. This method considers the relationships between features.

**Iterative Imputation (MICE)**: Apply Multiple Imputation by Chained Equations, which models each feature with missing values as a function of other features in a round-robin fashion. This iterative approach captures complex relationships between variables.

**Forward Fill and Backward Fill**: For time-series data or ordered datasets, use forward fill (propagate last valid observation forward) or backward fill (use next valid observation to fill backward) to maintain temporal consistency.

**Constant Value Imputation**: For specific columns, impute missing values with a constant value that makes sense in the domain context (e.g., 0 for missing counts, "Unknown" for missing categories).

These advanced imputation techniques are crucial for preserving data relationships and improving the quality of your dataset when simple imputation methods are insufficient.

### Task 4: Creating Interaction Features
You are provided with a dataset containing multiple numerical features. Your task is to create interaction features that capture relationships between existing features:

**Multiplication Interactions**: Create new features by multiplying pairs of features together. For example, if you have `height` and `width`, create a new feature `area = height * width`.

**Division Interactions**: Create ratio features by dividing one feature by another. For example, create a `price_per_square_foot = price / area` feature.

**Polynomial Features**: Generate polynomial features up to degree 2 or 3. For example, if you have a feature `x`, create `x²` and `x³`.

Interaction features can help capture non-linear relationships and improve model performance, especially for linear models.

### Task 5: Multi-Class Label Encoding with Ordinal Relationships
You are provided with a dataset containing categorical features with ordinal relationships (e.g., education level: "High School" < "Bachelor's" < "Master's" < "PhD"). Your task is to encode these features while preserving their ordinal nature:

**Ordinal Encoding**: Map categorical values to integers that reflect their order. For example:
- "High School" → 0
- "Bachelor's" → 1
- "Master's" → 2
- "PhD" → 3

**Handling New Categories**: For categories that might appear in the test set but not in the training set, assign a default value (e.g., -1 or the most common category).

**Validation**: Ensure that the encoding preserves the natural order and that the model can interpret the ordinal relationship correctly.

This task is important for features where the order matters, as it helps the model understand the hierarchical relationship between categories.

### Task 6: Time-Based Feature Engineering
You are provided with a dataset containing datetime information (e.g., transaction timestamps, birth dates). Your task is to extract meaningful temporal features that can improve model performance:

**Extract Date Components**: From a datetime column, extract features such as:
- Year, month, day, day of week, hour, minute
- Quarter (Q1, Q2, Q3, Q4)
- Is weekend (binary flag)
- Is month end/start (binary flags)

**Create Cyclic Features**: For periodic features like hour of day or month, create sine and cosine transformations to capture their cyclic nature. For example:
- `hour_sin = sin(2π × hour / 24)`
- `hour_cos = cos(2π × hour / 24)`

**Calculate Time Differences**: Create features representing time elapsed since a reference date (e.g., days since account creation, age in years).

These temporal features are critical for time-series analysis and can significantly improve model accuracy for time-dependent data.

### Task 7: Handling Imbalanced Data with Resampling Techniques
You are provided with a classification dataset where one class significantly outnumbers the others (imbalanced dataset). Your task is to address this imbalance using various resampling techniques:

**Random Oversampling**: Randomly duplicate samples from the minority class until the class distribution is balanced.

**Random Undersampling**: Randomly remove samples from the majority class to balance the distribution. Be cautious as this can lead to information loss.

**SMOTE (Synthetic Minority Over-sampling Technique)**: Generate synthetic samples for the minority class by interpolating between existing minority class samples. This creates new, realistic samples rather than just duplicating existing ones.

**Combination Approach**: Apply SMOTE followed by undersampling of the majority class (SMOTE + Tomek Links or SMOTE + ENN) to achieve a balanced dataset while preserving important information.

This task is crucial for classification problems where class imbalance can lead to poor model performance, as most algorithms are biased toward the majority class.