# Streamlit Application Plan: Production Decision-Support System

## 1. Application Objective
The goal is to transform the Prescriptive Analytics notebooks into a professional, interactive software tool. The app will allow production managers to enter current resource allocations and target quantities, then receive an immediate, AI-driven recommendation on how to adjust resources to hit their production goals.

---

## 2. User Workflow (The UX)
The application will follow a linear, three-step process:

### Step 1: Configuration & Input
The user provides the specific context for the production run:
- **Product Selection**: A searchable dropdown of `item_name` (loaded from the dataset).
- **Current State**: Input fields for existing `no_of_workers`, `working_hours`, and `machine_count`.
- **Production Goal**: A numerical input for the `planned_qty`.
- **Contextual Data**: The app automatically fetches associated `factory_unit`, `product_family`, `grade`, etc., based on the selected item.

### Step 2: Baseline Analysis (The "Current Reality")
Once the user clicks "Analyze", the app runs the **Notebook 6 logic**:
- **Prediction**: It calculates the `predicted_actual_qty` based on current resources.
- **Gap Detection**: It calculates the difference between the prediction and the plan.
- **State Alert**: The app displays a prominent status badge:
    - 🔴 **Overproduction**: "Predicted output exceeds plan by X units."
    - 🟡 **Underproduction**: "Predicted output is below plan by X units."
    - 🟢 **Balanced**: "Current resources are aligned with the plan."

### Step 3: Prescriptive Recommendation
If the state is not "Balanced," the app triggers the **Notebook 8 Delta Optimizer**:
- **Recommendation Table**: Displays the Top-3 resource configurations that hit the target.
- **Change Analysis**: Instead of just showing totals, it shows the **$\Delta$ (Change)**:
    - *Example: "Workers: 15 $\rightarrow$ 13 (Reduce by 2)"*
    - *Example: "Hours: 600 $\rightarrow$ 648 (Increase by 48)"*
- **Confidence Score**: Displays the predicted gap for the recommended setting.

---

## 3. Technical Architecture

### 3.1 Backend Logic (`app_engine.py`)
To keep the app fast, we will move the notebook logic into a separate Python module:
- `load_model()`: Singleton function to load `production_model.joblib`.
- `get_baseline_prediction()`: Logic from Notebook 6.
- `get_prescriptive_recommendations()`: The vectorized `DeltaOptimizer` from Notebook 8.
- `validate_constraints()`: The guardrails from Notebook 7.

### 3.2 Frontend Interface (`app.py`)
- **Framework**: Streamlit.
- **State Management**: Use `st.session_state` to remember the baseline prediction while the user explores different "what-if" scenarios.
- **Visualizations**:
    - **KPI Cards**: Displaying Planned vs. Predicted qty.
    - **Comparison Bar Chart**: Showing the resource difference between "Current" and "Recommended" states.

---

## 4. Development Roadmap

### Phase 1: Core Integration
- [ ] Create `app_engine.py` and migrate all logic from Notebooks 6, 7, and 8.
- [ ] Implement basic Streamlit inputs and model loading.

### Phase 2: UI/UX Enhancement
- [ ] Implement the "Status Badge" system (Red/Yellow/Green).
- [ ] Build the recommendation table with $\Delta$ (change) calculations.
- [ ] Add data validation (e.g., preventing negative numbers in inputs).

### Phase 3: Visualization & Polish
- [ ] Add Plotly/Altair charts to visualize the resource shift.
- [ ] Implement a "Download Report" button to export the recommendation as a CSV/PDF.
- [ ] Final end-to-end testing with real dataset scenarios.

---

## 5. Success Metrics
The app is considered successful if:
1. **Latency**: The transition from "Analyze" to "Recommendation" takes $< 2$ seconds.
2. **Accuracy**: The recommended resource mix results in a prediction within $\pm 2$ units of the target.
3. **Usability**: A manager can determine the necessary resource adjustment in under 30 seconds.
