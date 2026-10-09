# Machine Learning–Driven Production Forecasting and Prescriptive Analytics

## Project Overview

This project focuses on developing a machine learning–driven predictive and prescriptive analytics framework to forecast weekly production target achievement in the shoe manufacturing division of Ceylon Leather Products Manufacturers Ltd (DI Leather).

The framework uses historical production data and operational characteristics to predict actual production quantity. These predictions can support production planning by helping decision-makers understand expected production output and identify potential gaps between planned and actual production.

The project combines data preprocessing, exploratory data analysis, feature engineering, regression model evaluation, and production prediction into a structured machine learning workflow.

## Project Objectives

- Analyse historical production data to understand production patterns and operational factors.
- Clean and prepare the dataset for machine learning.
- Perform exploratory data analysis (EDA) to identify trends, distributions, and relationships among production variables.
- Engineer meaningful features from available operational data.
- Train and compare multiple regression models to predict actual production quantity.
- Select a suitable model based on performance on a chronological test dataset.
- Develop a prediction workflow for estimating the production output of a new production order.
- Use predicted production output to support production target-achievement analysis and production-planning decisions.

## Project Workflow

The project follows these five main stages.

### 1. Data Preprocessing

The raw production dataset is inspected and prepared for analysis. This stage includes checking data quality, examining missing values and duplicate records, correcting data types, creating relevant categorical features, and applying the implemented outlier-handling procedure.

### 2. Exploratory Data Analysis (EDA)

Exploratory analysis is conducted to understand production quantity distributions, temporal patterns, categorical production characteristics, numerical relationships, and potential operational factors associated with production output.

### 3. Feature Engineering and Dataset Preparation

Additional features are created to represent operational relationships, including machine-to-worker ratio and average working hours per worker. The dataset is then prepared for model training by selecting relevant predictors, excluding inappropriate or potentially leakage-related variables, and retaining the chronological information required for time-aware evaluation.

### 4. Model Selection and Training

Multiple regression algorithms are evaluated using a chronological training and testing split. The models considered include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- XGBoost Regressor

Linear Regression is used as a baseline model. The candidate models are compared using R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

Gradient Boosting is selected as the best-performing model among the evaluated configurations based on its test-set performance.

### 5. Production Prediction

The selected trained model is saved as a reusable machine learning pipeline. A prediction function accepts the required characteristics of a new production order, recreates the engineered features, and estimates the expected actual production quantity.

The predicted quantity can subsequently be compared with a planned production target to calculate business measures such as expected target achievement and production gap.

## Model Evaluation

The models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| R² | Measures how well the model explains variation in actual production quantity. |
| MAE | Measures the average absolute difference between predicted and actual production quantities. |
| RMSE | Measures prediction error while penalising larger errors more strongly. |

The selected Gradient Boosting model achieved the following results on the chronological test dataset:

| Metric | Test Result |
|---|---:|
| R² | 0.791 |
| MAE | 4.577 |
| RMSE | 6.000 |

The test R² of 0.791 indicates that the model explains approximately 79.1% of the variation in actual production quantity within the test dataset. The MAE indicates an average absolute prediction error of approximately 4.58 shoes per record, while the RMSE is approximately 6.00 shoes.

These results describe performance on the evaluated test period and should not be interpreted as a guarantee of future prediction accuracy.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** pandas, NumPy
- **Machine Learning:** scikit-learn, XGBoost
- **Categorical Feature Encoding:** One-hot encoding and target encoding
- **Data Visualisation:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook and Visual Studio Code
- **Version Control:** Git and GitHub

## Repository Structure

```text
production-DI/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Selection_and_Training.ipynb
│   └── 05_Prediction.ipynb
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── production_model.joblib
│
└── reports/
```

*This structure represents the recommended organisation of the repository. The folders and files should match the actual contents of the GitHub repository.*

## How to Use the Project

### Prerequisites

- Python installed on your computer.
- Jupyter Notebook support or Visual Studio Code with the Jupyter extension.
- The required Python libraries.
- Access to the relevant dataset files and saved model, where applicable.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/KGP-Pramodith-lang/Production-optimization.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Production-optimization
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Open the project in Visual Studio Code or Jupyter Notebook.

### Execution Order

Run the notebooks in the following order:

1. `01_Preprocessing.ipynb`
2. `02_EDA.ipynb`
3. `03_Feature_Engineering.ipynb`
4. `04_Model_Selection_and_Training.ipynb`
5. `05_Prediction.ipynb`

Ensure that the dataset paths, output paths, and saved model paths match the repository structure before executing the notebooks. The prediction notebook requires the saved model pipeline generated during model training.

## Expected Output

The prediction workflow produces an estimated actual production quantity for a new production order based on its supplied operational characteristics.

When a planned quantity is available, the prediction can also support downstream calculations such as:

- **Predicted production quantity:** the model's estimate of actual output.
- **Expected achievement ratio:** predicted production quantity divided by planned quantity.
- **Expected production gap:** planned quantity minus predicted production quantity.

These measures can help support production planning and resource-related decisions. They are estimates derived from the prediction and should be interpreted alongside operational knowledge.

## Limitations

- Model performance depends on the quality, representativeness, and availability of the historical production data.
- Changes in production processes, product mix, resource availability, or operating conditions may affect future prediction accuracy.
- Predictions are estimates rather than guaranteed production outcomes.
- Model evaluation results apply to the selected historical test period and may differ under future production conditions.
- Recommendations about changing resources or production plans require further validation and should not be based on predictions alone.

## Future Work

Potential extensions include:

- Further validation using newly collected production data.
- Model hyperparameter tuning and performance monitoring.
- Integration of prediction outputs into a production-planning dashboard.
- Development and evaluation of prescriptive analytics to compare production scenarios and support resource allocation.
- Assessment of the practical value of model-assisted production decisions with domain experts.

## Data Confidentiality

This project relates to production data from a manufacturing organisation. Dataset files, model artefacts, and other company-related information should only be shared publicly when the appropriate permission has been obtained. Sensitive or confidential information should not be committed to a public repository.

## Acknowledgements

This project was developed as an academic data science and business analytics project focused on applying machine learning to manufacturing production forecasting and decision support.


