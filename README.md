# DI Leather Production Forecasting System

A machine learning-driven predictive and prescriptive analytics framework for forecasting weekly production target achievement in the shoe manufacturing division of DI Leather.

## 📌 Project Overview

This project implements a data-driven approach to predict the actual production quantity (`actual_qty`) of shoes based on resource availability and operating conditions. By leveraging historical production data, the system helps the manufacturing division move from reactive monitoring to proactive forecasting and resource planning.

The core objective is to build a reliable predictive model that can estimate production output given specific inputs like workforce size, machine count, and working hours, providing a foundation for prescriptive analytics and optimized resource allocation.

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/              # Original weekly production datasets
│   └── processed/        # Preprocessed and engineered feature sets
├── models/               # Saved trained model pipelines (.joblib)
├── notebooks/            # Step-by-step analysis and development
│   ├── 01_Preprocessing.ipynb           # Data cleaning and standardization
│   ├── 02_EDA.ipynb                      # Exploratory Data Analysis
│   ├── 03_Feature_Engineering.ipynb      # Feature creation and selection
│   ├── 04_Model_Selection_and_Training.ipynb # Model tuning and evaluation
│   └── 05_Prediction.ipynb               # Prediction workflow implementation
├── prescriptive/        # Prescriptive analytics and resource optimization
│   ├── prediction.ipynb
│   └── prescriptive.ipynb
└── reports/             # Research proposals and documentation
```

## 🛠️ Technical Workflow

### 1. Data Preprocessing & EDA
- **Cleaning:** Handling of duplicates, missing values, and outlier detection using the Interquartile Range (IQR) method on production rates.
- **Standardization:** Conversion of dates and boolean flags to ensure data integrity.
- **Exploration:** Analysis of temporal patterns (weekly, monthly, yearly) and categorical drivers (factory unit, product family, shoe size).

### 2. Feature Engineering
- **Resource Ratios:** Creation of `machine_worker_ratio` and `avg_hours_per_worker` to capture resource intensity.
- **Leakage Prevention:** Strict removal of post-production metrics (e.g., achievement rates, production gaps) to ensure predictors are available at the time of forecasting.
- **Encoding Strategy:** 
  - **Standard Scaling** for numerical features.
  - **One-Hot Encoding** for low-cardinality categories (e.g., `factory_unit`, `grade`).
  - **Target Encoding** for high-cardinality features (e.g., `item_name`).

### 3. Predictive Modeling
- **Algorithm Comparison:** Evaluation of seven regression models, including Linear Regression (baseline), Ridge, Lasso, Random Forest, Gradient Boosting, Extra Trees, and XGBoost.
- **Validation Strategy:** Use of `TimeSeriesSplit` (5-fold) to preserve chronological order and prevent data leakage from future observations.
- **Model Selection:** **XGBoost** was selected as the optimal model based on the lowest cross-validation Mean Absolute Error (MAE).
- **Performance Metrics (Test Set):**
  - **R²:** ~0.795 (explains ~80% of production variance).
  - **MAE:** ~4.507 units (average error of approx. 4.5 shoes).
  - **RMSE:** ~5.920 units.

### 4. Prediction & Prescriptive Analysis
- **Deployment:** The final pipeline (preprocessing + model) is saved as a `.joblib` file for rapid inference.
- **Inference:** Ability to input a new production order and receive a rounded integer prediction of actual output.
- **Prescriptive Layer:** Comparing predictions against planned targets to recommend resource adjustments.

## 🚀 Getting Started

### Prerequisites
Ensure you have the required Python environment. While `requirements.txt` is provided, the project primarily relies on:
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`
- `matplotlib`, `seaborn`

### Running the Pipeline
To replicate the results, execute the notebooks in sequential order:
1. `01_Preprocessing.ipynb` $\rightarrow$ `02_EDA.ipynb` $\rightarrow$ `03_Feature_Engineering.ipynb` $\rightarrow$ `04_Model_Selection_and_Training.ipynb` $\rightarrow$ `05_Prediction.ipynb`.

## 📊 Key Findings
- **Primary Drivers:** `machine_count` and `working_hours` are the most significant predictors of production output.
- **Operational Insight:** The relationship between resources and output is largely additive, though boosted tree models provide a modest improvement over linear baselines.
- **Bottleneck Impact:** The `bottleneck_flag` significantly associates with lower average production output.
