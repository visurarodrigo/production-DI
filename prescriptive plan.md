# Prescriptive Analytics Engine: Resource Optimization Plan

## 1. Project Philosophy
The system follows a "Baseline-to-Target" workflow. Instead of simply searching for any configuration that hits a target, the system analyzes the *current* resource allocation and suggests the *minimum necessary adjustment* to align production with the plan.

### The Logic Flow:
1. **Baseline Prediction**: Input current resources $\rightarrow$ Model predicts `predicted_actual_qty`.
2. **Gap Classification**:
    - $\text{predicted\_actual\_qty} > \text{planned\_qty}$ $\rightarrow$ **Overproduction** $\rightarrow$ Strategy: **Resource Reduction**.
    - $\text{predicted\_actual\_qty} < \text{planned\_qty}$ $\rightarrow$ **Underproduction** $\rightarrow$ Strategy: **Resource Expansion**.
    - $\text{predicted\_actual\_qty} \approx \text{planned\_qty}$ $\rightarrow$ **Balanced** $\rightarrow$ Strategy: **No Change**.
3. **Prescriptive Optimization**: Search for a new configuration that minimizes the gap while keeping changes relative to the baseline.

---

## 2. Technical Architecture
The engine is developed and validated using a series of **Jupyter Notebooks**, allowing for iterative testing and immediate verification of recommendations.

### 2.1 Notebook Structure
| Notebook | Name | Responsibility |
| :--- | :--- | :--- |
| **Notebook 6** | `06_Prescriptive_Feature_Engineering.ipynb` | Baseline Prediction: Predicts actual output from current resources and classifies the gap (Over/Under/Balanced). |
| **Notebook 7** | `07_Prescriptive_Constraints_Validation.ipynb` | Operational Guardrails: Defines min/max resource limits and actionable change steps (e.g., 1 worker, 8 hours). |
| **Notebook 8** | `08_Prescriptive_Optimizer_Core.ipynb` | Delta Optimizer: A high-performance vectorized engine that suggests the smallest resource change to hit the target. |
| **Notebook 9** | `09_Prescriptive_Final_Validation.ipynb` | Systematic Validation: Stress tests the directional logic across all three production states. |

---

## 3. Detailed Implementation Roadmap

### Step 1: Baseline & Gap Analysis (`Notebook 6`)
- **Baseline Prediction**: Use the Phase 1 model to predict `actual_qty` based on currently available resources.
- **Gap Classification**: Compare prediction to `planned_qty` and categorize as Overproduction, Underproduction, or Balanced.
- **Trigger Logic**: Establish the logic that determines the search direction for the optimizer.

### Step 2: Resource Boundary Definitions (`Notebook 7`)
- **Absolute Bounds**: Set minimum and maximum allowable limits for workers, hours, and machines to prevent impossible suggestions.
- **Actionable Steps**: Define the increments for changes (e.g., workers change by 1, hours by 8) to ensure recommendations are realistic for factory managers.
- **Feasibility Validator**: Implement the `is_feasible` function to filter the search space.

### Step 3: The Delta Optimizer (`Notebook 8`)
- **Directional Search**: 
    - If **Underproduction** $\rightarrow$ Search only configurations where $\text{resources} \ge \text{baseline}$.
    - If **Overproduction** $\rightarrow$ Search only configurations where $\text{resources} \le \text{baseline}$.
- **Vectorized Simulation**: Use Pandas to simulate thousands of resource combinations in a single model call for instant results.
- **Proximity Scoring**: Rank recommendations by the smallest total deviation from the current baseline that successfully hits the target.

### Step 4: End-to-End Validation (`Notebook 9`)
- **Underproduction Scenario**: Target $\gg$ Baseline $\rightarrow$ Verify AI suggests increasing resources.
- **Overproduction Scenario**: Target $\ll$ Baseline $\rightarrow$ Verify AI suggests reducing resources.
- **Stability Scenario**: Target $\approx$ Baseline $\rightarrow$ Verify AI suggests "No change needed".

---

## 4. Technical Requirements & Constraints

### 4.1 Model Integrity
- **No Data Leakage**: The `planned_qty` is used only for gap analysis and scoring, never as an input feature for the ML model.
- **Consistent Preprocessing**: Use the `production_model.joblib` pipeline to ensure categorical encoding and scaling match the training phase.

### 4.2 Performance
- **Vectorization**: Avoid Python loops for predictions; use NumPy/Pandas arrays to perform batch predictions for the entire search space.
- **Search Space**: Use defined "steps" (e.g., 8-hour blocks) to keep the number of combinations manageable.

---

## 5. Transition to Streamlit App (Next Phase)
The logic from these notebooks will be wrapped into a professional Decision-Support Application.

### App Architecture:
1. **Input Panel**: User selects the shoe $\rightarrow$ enters current resources $\rightarrow$ enters planned qty.
2. **Analysis Engine**: Runs the baseline prediction $\rightarrow$ detects state (Over/Under) $\rightarrow$ runs the Delta Optimizer.
3. **Result Dashboard**: 
    - **Status Alert**: "Warning: Overproduction detected (Predicted 66 vs Planned 30)."
    - **Prescription**: "Recommendation: Reduce workers by 2 to align with target."
    - **Visuals**: Comparison charts of Baseline vs. Recommended configurations.
