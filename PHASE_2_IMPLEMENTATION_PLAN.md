# Phase 2: Prescriptive Analytics & Production Decision-Support Dashboard Implementation Plan

## 1. Project Overview
The goal of Phase 2 is to transition from **Predictive Analytics** (predicting what will happen) to **Prescriptive Analytics** (recommending what to do). We will build an optimization system that uses the trained Phase 1 model to suggest optimal resource configurations (workers, hours, machines) to achieve a specific `planned_qty`.

### Core Logic
- **Input:** Planned Quantity, Product Context (Item, Unit, etc.), and Baseline Resources.
- **Engine:** The saved Voting Ensemble model (`production_model.joblib`).
- **Process:** The system predicts the baseline output. If the gap between planned and predicted is outside a tolerance band, the optimizer searches for feasible resource combinations that minimize this gap and resource cost.
- **Output:** A ranked list of recommended configurations presented via a Streamlit dashboard.

---

## 2. Full Implementation Roadmap (0 to Full)

### Step 1: Foundation & Service Layer (`src/`)
Before building the UI, we must create the backend logic to handle predictions and optimization.

#### 2.1 `src/feature_builder.py`
**Purpose:** Ensure consistency between training data and real-time predictions.
- **Functionality:**
    - Take raw inputs (workers, hours, machines, item_name, etc.).
    - Calculate engineered features: `machine_worker_ratio` and `avg_hours_per_worker`.
    - Return a DataFrame in the exact format expected by the saved pipeline.

#### 2.2 `src/prediction_service.py`
**Purpose:** A wrapper around the saved model.
- **Functionality:**
    - Load `models/production_model.joblib` once.
    - Provide a `predict(features)` function that returns the `predicted_actual_qty`.

#### 2.3 `src/constraints.py`
**Purpose:** Define the boundaries of "feasible" recommendations.
- **Functionality:**
    - Define min/max ranges for `no_of_workers`, `working_hours`, and `machine_count`.
    - Implement validation logic to reject candidates that exceed physical or company limits.

#### 2.4 `src/prescriptive_optimizer.py`
**Purpose:** The "brain" of Phase 2.
- **Functionality:**
    - **Baseline Calculation:** Predict output for current settings.
    - **Exhaustive Search:** Generate a grid of feasible combinations of workers, hours, and machines.
    - **Scoring Engine:** For each candidate, predict output $\hat{y}$ and calculate a score:
      $Score = w_{gap} \cdot |Target - \hat{y}| + w_{res} \cdot \text{ResourceCost}$
    - **Ranking:** Return candidates sorted by the lowest score.

---

### Step 2: The Interactive Dashboard (`app.py`)
Build the user interface using **Streamlit**.

#### 2.1 Input Section
- **Mode Selection:** "Future Planning" vs. "Historical Review".
- **Context Selection:** Dropdowns for `item_name`, `factory_unit`, `product_family`, `size_id`, `grade`, `order_category`.
- **Target Setting:** Numerical input for `planned_qty`.
- **Baseline Setting:** Current `no_of_workers`, `working_hours`, `machine_count`.

#### 2.2 Analysis Section
- **Baseline Prediction:** Display the predicted `actual_qty` for current settings.
- **Gap Analysis:** 
    - Calculate $Gap = planned\_qty - predicted\_qty$.
    - Classify as: **Within Tolerance**, **Underproduction**, or **Overproduction**.
- **Visuals:** Gauge chart or KPI cards showing the gap.

#### 2.3 Prescriptive Section (Conditional)
- Only triggered if the gap is outside the tolerance band.
- **Recommendations Table:** 
    - Predicted Quantity | Gap | Suggested Workers | Suggested Hours | Suggested Machines | Resource Change ($\Delta$).
- **Warnings:** Explicit labels stating "Estimates only - requires planner review."

---

## 3. File Structure to Create

```text
project-root/
├── models/
│   └── production_model.joblib       # (Existing) Saved Phase 1 Model
├── data/
│   └── processed/
│       └── final_features.csv        # (Existing) For reference/validation
├── src/
│   ├── __init__.py
│   ├── feature_builder.py           # NEW: Logic to build model input rows
│   ├── prediction_service.py        # NEW: Model loading and prediction wrapper
│   ├── constraints.py               # NEW: Feasibility ranges and rules
│   └── prescriptive_optimizer.py     # NEW: Search and scoring logic
├── app.py                           # NEW: Streamlit Dashboard UI
├── requirements.txt                 # UPDATE: Add 'streamlit'
└── outputs/                          # For logging recommendations
```

---

## 4. Detailed Technical Requirements

### 4.1 Model Input Guardrails
- **No Leakage:** `planned_qty` and `actual_qty` MUST NOT be inputs to the model.
- **Consistent Preprocessing:** The `production_model.joblib` contains the pipeline (StandardScaler, OneHotEncoder). We must provide the raw categorical strings and numbers; the pipeline handles the rest.
- **Dynamic Features:** `machine_worker_ratio` and `avg_hours_per_worker` must be recalculated for *every* candidate configuration during optimization.

### 4.2 Optimization Logic (Version 1)
- **Method:** Exhaustive Enumeration (Grid Search).
- **Search Space:** 
    - Workers: [Min, ..., Max] (Step: 1)
    - Hours: [Min, ..., Max] (Step: 4 or 8)
    - Machines: [Min, ..., Max] (Step: 1)
- **Feasibility Check:** Reject candidates that violate rules in `constraints.py`.

### 4.3 Dashboard UX Principles
- **Business Labels:** Use "Staffing Level" instead of `no_of_workers`.
- **Transparency:** Show why a recommendation was made (e.g., "Reduces shortfall by 5 units while adding 2 workers").
- **Caution:** High-visibility disclaimer: *"Model-based estimates. Not a guarantee of outcome."*

---

## 5. Validation Plan
1. **Predictive Check:** Compare `app.py` baseline predictions against `Notebook 5` results for the same inputs.
2. **Constraint Check:** Verify that the optimizer never recommends 0 workers or 100 machines.
3. **Scenario Testing:**
    - **Case A (Under):** Plan = 100, Predicted = 80 $\rightarrow$ Expect recommendations to increase resources.
    - **Case B (Over):** Plan = 100, Predicted = 120 $\rightarrow$ Expect recommendations to decrease resources.
    - **Case C (OK):** Plan = 100, Predicted = 101 $\rightarrow$ Expect "Within Tolerance" message.
