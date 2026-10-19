# Prescriptive Analytics Engine Implementation Plan

## 1. Overview
The objective is to build a standalone **Prescriptive Analytics Engine**. This system moves beyond predicting production outcomes to recommending the optimal resource configuration (workers, hours, and machines) required to meet a specific production target (`planned_qty`).

The engine utilizes the pre-trained Phase 1 Voting Ensemble model to simulate various resource scenarios and identifies the configuration that minimizes the gap between predicted and planned quantity while optimizing for resource cost.

---

## 2. Technical Architecture
The engine will be developed and validated using a series of **Jupyter Notebooks**, allowing for iterative testing, visualization of the search space, and immediate verification of recommendations.

### 2.1 Notebook Structure
| Notebook | Name | Responsibility |
| :--- | :--- | :--- |
| **Notebook 6** | `prescriptive_feature_engineering.ipynb` | Developing the logic to transform raw resource inputs into model-ready features. |
| **Notebook 7** | `prescriptive_constraints_validation.ipynb` | Defining and testing the feasibility bounds for resources. |
| **Notebook 8** | `prescriptive_optimizer_core.ipynb` | Implementing the Grid Search, scoring function, and ranking logic. |
| **Notebook 9** | `prescriptive_final_validation.ipynb` | End-to-end scenario testing (Under/Overproduction cases). |

---

## 3. Detailed Implementation Roadmap

### Step 1: Feature Engineering (`Notebook 6`)
To ensure the model produces accurate predictions, the inputs must exactly match the training data distribution.
- **Raw Input Handler:** Create functions that accept `(workers, hours, machines, item_name, unit, family, size, grade, category)`.
- **Derived Feature Calculation:** Implement the specific formulas used in Phase 1:
    - `machine_worker_ratio` = $\text{machines} / \text{workers}$
    - `avg_hours_per_worker` = $\text{hours} / \text{workers}$
- **Pipeline Integration:** Test the transformation by passing a sample row through the `production_model.joblib` pipeline to verify output shapes.

### Step 2: Constraint Definitions (`Notebook 7`)
The optimizer must only suggest configurations that are physically and operationally possible.
- **Resource Bounds:** Define a configuration dictionary for min/max limits:
    - `no_of_workers`: $[Min, Max]$
    - `working_hours`: $[Min, Max]$
    - `machine_count`: $[Min, Max]$
- **Feasibility Logic:** Create a validator `is_feasible(config)` that returns a boolean based on these bounds and any specific business rules.
- **Edge Case Testing:** Test the validator with extreme values to ensure it correctly flags invalid configurations.

### Step 3: The Optimization Engine (`Notebook 8`)
This is the core logic that performs the prescriptive analysis.
- **Search Space Generation:** Implement an exhaustive grid search that iterates through all combinations of feasible workers, hours, and machines.
- **Simulation Loop:**
    1. Generate candidate configuration.
    2. Use logic from Notebook 6 to build features.
    3. Predict $\hat{y}$ (predicted quantity) using the saved model.
- **Multi-Objective Scoring:** Implement a scoring function to rank the "best" configuration:
    $$\text{Score} = (w_{gap} \cdot |\text{Target} - \hat{y}|) + (w_{res} \cdot \text{ResourceCost})$$
    - $w_{gap}$: Weight for production accuracy.
    - $w_{res}$: Weight for resource economy.
- **Ranking:** Extract the Top-N configurations with the lowest scores and visualize the trade-off between cost and accuracy.

### Step 4: End-to-End Validation (`Notebook 9`)
Final testing to ensure the prescriptive logic is robust across different production targets.
- **Underproduction Scenario:**
    - Input: Target = 100, Baseline Prediction = 70.
    - Expectation: Optimizer recommends increasing workers or hours.
- **Overproduction Scenario:**
    - Input: Target = 100, Baseline Prediction = 130.
    - Expectation: Optimizer recommends reducing resources to minimize cost while staying above 100.
- **Stability Scenario:**
    - Input: Target = 100, Baseline Prediction = 101.
    - Expectation: Optimizer identifies current settings as optimal.

---

## 4. Technical Requirements & Constraints

### 4.1 Model Integrity
- **No Data Leakage:** The target `planned_qty` must never be an input to the model; it is only used by the optimizer to calculate the gap.
- **Pipeline Handling:** The input to the model must be the raw categorical strings and numbers; the `joblib` pipeline (which includes the `StandardScaler` and `OneHotEncoder`) must handle the transformation.

### 4.2 Search Complexity
- **Grid Resolution:** To maintain performance, the search space should use reasonable steps:
    - Workers: Step of 1.
    - Hours: Step of 4 or 8.
    - Machines: Step of 1.

---

## 5. Validation & Testing Plan

### 5.1 Unit Testing
- **Feature Test:** Verify that `feature_builder` produces the exact same feature values as the original Phase 1 training notebooks.
- **Constraint Test:** Verify that the optimizer rejects any configuration exceeding the `constraints.py` limits.

### 5.2 Scenario Validation
- **Underproduction Scenario:**
    - Input: Target = 100, Baseline Prediction = 70.
    - Expectation: Optimizer recommends increasing workers or hours.
- **Overproduction Scenario:**
    - Input: Target = 100, Baseline Prediction = 130.
    - Expectation: Optimizer recommends reducing resources to minimize cost while staying above 100.
- **Stability Scenario:**
    - Input: Target = 100, Baseline Prediction = 101.
    - Expectation: Optimizer identifies current settings as optimal.
