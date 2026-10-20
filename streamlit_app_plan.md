# DI Leather Streamlit Dashboard Plan

## 1. Purpose

Build a company-facing Streamlit dashboard for the DI Leather shoe-manufacturing production project. The dashboard uses the trained machine-learning pipeline to predict weekly actual production quantity and provides decision-support recommendations for reviewing resource alternatives against a planned production quantity.

The application is a decision-support prototype. It does not automatically change factory operations, and its recommendations must be reviewed by production planners.

## 2. Project files and dependencies

Expected project structure:

```text
production-DI/
├── app.py
├── assets/
│   └── di_leather_shoes.png       # optional branding image
├── data/
│   └── processed/
│       └── final_features.csv
├── models/
│   └── production_model.joblib
├── notebooks/
├── outputs/
└── requirements.txt
```

Main dependencies:

- `streamlit` — interactive web dashboard
- `pandas` — data loading and tables
- `numpy` — numeric validation and calculations
- `joblib` — loading the saved model pipeline

Install dependencies from the project root, for example:

```powershell
python -m pip install streamlit pandas numpy joblib
```

Run the application from the project root:

```powershell
python -m streamlit run app.py
```

## 3. Prediction model and features

- Model file: `models/production_model.joblib`
- Historical feature data: `data/processed/final_features.csv`
- Prediction target: `actual_qty`
- Input features, in model order:
  1. `financial_month`
  2. `week_no`
  3. `week_of_month`
  4. `factory_unit`
  5. `product_family`
  6. `item_name`
  7. `size_id`
  8. `grade`
  9. `rate`
  10. `no_of_workers`
  11. `working_hours`
  12. `machine_count`
  13. `bottleneck_flag`
  14. `order_category`
  15. `machine_worker_ratio`
  16. `avg_hours_per_worker`

The dashboard must preserve these feature names and the trained model's expected order. It recalculates the engineered features as:

- `machine_worker_ratio = machine_count / no_of_workers`
- `avg_hours_per_worker = working_hours / no_of_workers`

`working_hours` means total labour-hours across all workers for the week, not hours per worker.

## 4. Dashboard design

### Branding and visual style

- Use a dark navy and blue company-facing theme.
- Display DI Leather branding in the sidebar.
- Use a leather-shoe image as a hero visual when `assets/di_leather_shoes.png` exists.
- Use clean cards, readable typography, and clear spacing.
- Do not display internal model paths, feature counts, historical record counts, or technical model metadata as primary company-facing content.

### Sidebar

- DI Leather branding
- Production workspace navigation labels
- Short business-oriented description

### Main sections

1. **Hero header** — production-planning title and short explanation.
2. **Production context** — factory unit, product family, item, size, grade, order category, financial month, week number, and week of month.
3. **Production plan** — planned weekly quantity and acceptable difference from target.
4. **Current resources** — number of workers, total weekly labour-hours, number of machines, rate, and known bottleneck status.
5. **Prediction result** — planned quantity, predicted actual quantity, and predicted gap.
6. **Production status** — prominent colour-coded classification.
7. **Prescriptive recommendations** — current versus proposed resource values, predicted quantity, gap, and scenario table.
8. **Scenario export** — allow users to download the evaluated resource scenarios as CSV.

## 5. Target comparison and status logic

Use:

`predicted_gap = predicted_actual_qty - planned_quantity`

Let `tolerance` be the user-selected acceptable difference.

- **Underproduction:** `predicted_gap < -tolerance`
- **Overproduction:** `predicted_gap > tolerance`
- **Keep resources unchanged:** `abs(predicted_gap) <= tolerance`

The dashboard should explain that the gap is predicted actual quantity minus planned quantity. Negative values indicate a predicted shortfall; positive values indicate predicted excess.

The planned quantity must be a real target for the selected product and week. The previous value of 5,000 was a test input and must not be treated as a default production target. Quantity units (individual shoes, pairs, or another unit) must be confirmed and used consistently with the source data.

## 6. Input validation

- Restrict workers, machines, weekly labour-hours, and rate to the available historical minimum and maximum values in the prototype.
- Display invalid inputs clearly and disable prediction until the inputs pass validation.
- Require a positive planned quantity.
- Require a non-negative tolerance.
- Warn when the target is unusually large relative to historical actual production, since it may indicate an incorrect scale or unit.
- Keep product and order context fixed while evaluating resource alternatives.
- Ensure workers and machines are integer values and all numeric inputs are finite.

**Important limitation:** historical min/max values are provisional data-based guardrails, not verified factory capacity or policy limits. Replace them with approved operating limits from DI Leather before operational use.

## 7. Prescriptive scenario search

For a prototype, enumerate a manageable set of candidate values for:

- `no_of_workers`
- `working_hours`
- `machine_count`

For each candidate:

1. Keep product and order context fixed.
2. Update resource values.
3. Recalculate the engineered ratio features.
4. Call the saved model to predict `actual_qty`.
5. Calculate the gap from the planned quantity.
6. Rank candidates by whether they meet tolerance, then target closeness, with a temporary resource-use proxy as a tie-breaker.

The temporary proxy `workers + weekly labour-hours + machines` mixes different units and is not a valid monetary cost or validated efficiency measure. It must not be presented as a true cost objective. Replace it with company-approved labour, machine, overtime, and other costs when those data become available.

### Recommendation behaviour

- **Underproduction:** search for scenarios that reduce the predicted shortfall and show current versus proposed resources.
- **Overproduction:** search for scenarios closer to the target, potentially using fewer resources, and show whether the candidate is within tolerance.
- **Within tolerance:** recommend keeping resources unchanged; no automatic change is needed under the selected tolerance.
- If no candidate meets tolerance, show the best tested scenario and clearly state the remaining predicted shortfall or excess.
- Never imply that a model prediction proves a resource change will cause the predicted production result.

## 8. Result presentation

Show three key metrics:

- Planned quantity
- Predicted actual quantity
- Predicted gap

Use large, prominent status cards:

- Red: **UNDERPRODUCTION**
- Orange: **OVERPRODUCTION**
- Green: **KEEP RESOURCES UNCHANGED**

In the prescriptive panel, show:

- Current workers, weekly labour-hours, and machines
- Proposed workers, weekly labour-hours, and machines
- Changes from current resources
- Predicted output for the proposed scenario
- Gap from target and whether it meets tolerance
- A table of other evaluated scenarios, available in an expandable section
- CSV download for the evaluated candidate table

## 9. Safety and operational limitations

The dashboard is a prototype decision-support tool. Before using a recommendation, a production planner must confirm:

- Worker availability and approved staffing levels
- Compatible machine availability and maintenance status
- Shift, regular-hour, and overtime limits
- Material availability
- Order deadlines and scheduling requirements
- Quality requirements
- Actual operating costs and the company's preferred objective

The model is observational and trained on historical data; candidate predictions are estimates, not guaranteed outcomes or causal effects. The dashboard must not automatically implement production changes.

## 10. Testing checklist

- [ ] App starts from the project root using Streamlit.
- [ ] Model and CSV load successfully from the expected paths.
- [ ] All 16 model features are supplied in the correct order.
- [ ] Product selections and dependent options work.
- [ ] `working_hours` is treated as total weekly labour-hours.
- [ ] Invalid values are blocked and clearly explained.
- [ ] Planned quantity is a realistic value in the correct unit.
- [ ] Underproduction, overproduction, and within-tolerance cases are tested.
- [ ] Current and proposed resource scenarios are displayed correctly.
- [ ] The app handles cases where no candidate meets tolerance.
- [ ] Scenario CSV download works.
- [ ] Historical input bounds are replaced with factory-approved limits before operational use.
- [ ] Recommendations are reviewed by DI Leather production staff before use.

## 11. Future improvements

- Obtain factory-approved operating limits and actual availability data.
- Replace the temporary resource proxy with a transparent cost or resource objective agreed with the company.
- Include additional constraints such as compatible machines, shifts, overtime, materials, quality, and delivery dates.
- Validate recommendation quality with production experts and retrospective cases.
- Record planner decisions and compare predicted versus realised production.
- Add role-based access, logging, deployment security, and IT review before any wider company deployment.
