
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==========================================================
# 1. PAGE CONFIGURATION AND DESIGN
# ==========================================================

st.set_page_config(
    page_title="DI Leather | Production Intelligence",
    page_icon="👞",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Overall page */
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1d35 65%, #102747 100%);
    color: #edf4ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, p, label {
    color: #edf4ff;
}

h1 {
    font-size: 2.6rem !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 1.8rem !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 1.3rem !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #09172a;
    border-right: 1px solid #203d61;
}

/* Inputs */
[data-testid="stNumberInput"],
[data-testid="stSelectbox"] {
    background-color: #132a47;
    border-radius: 10px;
}

input, textarea {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #168aad, #2563eb);
    color: white;
    border: 0;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #0e7490;
    color: white;
    border: 1px solid #67e8f9;
}

/* Section panels */
.section-card {
    background: #102541;
    border: 1px solid #24496f;
    border-radius: 16px;
    padding: 22px;
    margin: 14px 0 22px 0;
}

.prescriptive-card {
    background: linear-gradient(135deg, #211944, #142c48);
    border: 2px solid #8b5cf6;
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
}

.status-card {
    border-radius: 16px;
    padding: 24px;
    margin: 15px 0;
    border: 2px solid;
}

.status-under {
    background: #351a27;
    border-color: #ff4d6d;
}

.status-over {
    background: #38251b;
    border-color: #fb923c;
}

.status-ok {
    background: #10372f;
    border-color: #34d399;
}

.status-title {
    font-size: 2.15rem;
    font-weight: 850;
    margin-bottom: 8px;
    line-height: 1.2;
}

.status-detail {
    font-size: 1.1rem;
}

/* Highlight validation errors */
[data-testid="stAlert"][kind="error"] {
    background-color: #431820;
    border: 2px solid #ff4d6d;
    color: #ffffff;
    border-radius: 12px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #112b49;
    border: 1px solid #285178;
    padding: 16px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #67e8f9;
    font-weight: 800;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #345477;
    border-radius: 10px;
}

/* Info messages */
[data-testid="stAlert"][kind="info"] {
    background: #102e49;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


st.title("👞 DI Leather Production Intelligence")
st.markdown(
    "Machine-learning prediction **+** prescriptive production planning"
)

st.caption(
    "Weekly production decision support | Predictions are estimates, "
    "and recommendations require operational validation."
)


# ==========================================================
# 2. FILE PATHS AND MODEL FEATURES
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "production_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_features.csv"

FEATURES = [
    "financial_month",
    "week_no",
    "week_of_month",
    "factory_unit",
    "product_family",
    "item_name",
    "size_id",
    "grade",
    "rate",
    "no_of_workers",
    "working_hours",
    "machine_count",
    "bottleneck_flag",
    "order_category",
    "machine_worker_ratio",
    "avg_hours_per_worker"
]


# ==========================================================
# 3. LOAD THE TRAINED MODEL AND DATA
# ==========================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Processed dataset not found: {DATA_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("The model or dataset could not be loaded.")
    st.code(str(exc))
    st.stop()


missing_columns = [
    column for column in FEATURES + ["actual_qty"]
    if column not in df.columns
]

if missing_columns:
    st.error(f"Required columns are missing: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The processed dataset is empty.")
    st.stop()

if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
    if model_features != FEATURES:
        st.error(
            "The feature order in the saved model differs from this app. "
            "Check the model's expected feature names and order."
        )
        st.stop()


# ==========================================================
# 4. DATA VALIDATION AND HISTORICAL LIMITS
# ==========================================================

def numeric_series(column):
    values = pd.to_numeric(df[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def limits_for(column):
    values = numeric_series(column)

    if values.empty:
        return None, None

    return float(values.min()), float(values.max())


# Provisional boundaries based on the observed dataset.
# Replace these with factory-approved limits when available.
WORKER_MIN, WORKER_MAX = limits_for("no_of_workers")
MACHINE_MIN, MACHINE_MAX = limits_for("machine_count")
HOUR_MIN, HOUR_MAX = limits_for("working_hours")
RATE_MIN, RATE_MAX = limits_for("rate")
WEEK_MIN, WEEK_MAX = 1, 53
WEEK_MONTH_MIN, WEEK_MONTH_MAX = 1, 5


def get_options(series):
    return sorted(series.dropna().unique().tolist(), key=str)


def median_value(series, fallback):
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if values.empty:
        return fallback

    return float(values.median())


def check_range(label, value, minimum, maximum):
    errors = []

    if not np.isfinite(float(value)):
        errors.append(f"{label} must be a valid number.")
    elif float(value) < minimum or float(value) > maximum:
        errors.append(
            f"{label} must be between {minimum:g} and {maximum:g}."
        )

    return errors


# Validate data used to define boundaries.
for label, low, high in [
    ("Workers", WORKER_MIN, WORKER_MAX),
    ("Machines", MACHINE_MIN, MACHINE_MAX),
    ("Working hours", HOUR_MIN, HOUR_MAX),
    ("Rate", RATE_MIN, RATE_MAX)
]:
    if low is None or high is None:
        st.error(f"Cannot determine valid limits for {label}.")
        st.stop()

if (df["actual_qty"].isna().any()
        or not np.isfinite(pd.to_numeric(
            df["actual_qty"], errors="coerce"
        )).all()):
    st.error("The actual_qty target contains invalid values.")
    st.stop()

if (pd.to_numeric(df["actual_qty"], errors="coerce") < 0).any():
    st.error("The actual_qty target contains negative values.")
    st.stop()


# ==========================================================
# 5. HELPER FUNCTIONS
# ==========================================================

def make_input(
    month, week, week_month,
    factory, family, item, size, grade,
    rate, workers, hours, machines, bottleneck, category
):
    row = {
        "financial_month": month,
        "week_no": int(week),
        "week_of_month": int(week_month),
        "factory_unit": factory,
        "product_family": family,
        "item_name": item,
        "size_id": size,
        "grade": grade,
        "rate": float(rate),
        "no_of_workers": int(workers),
        "working_hours": float(hours),
        "machine_count": int(machines),
        "bottleneck_flag": bool(bottleneck),
        "order_category": category,
        "machine_worker_ratio": float(machines) / float(workers),
        "avg_hours_per_worker": float(hours) / float(workers)
    }

    return pd.DataFrame([row], columns=FEATURES)


def predict_quantity(input_df):
    result = model.predict(input_df)
    prediction = float(np.asarray(result).reshape(-1)[0])

    if not np.isfinite(prediction) or prediction < 0:
        raise ValueError("The model returned an invalid prediction.")

    return prediction


def production_status(prediction, planned, tolerance):
    gap = prediction - planned

    if abs(gap) <= tolerance:
        return "KEEP RESOURCES AS THEY ARE", "on_target"

    if gap < 0:
        return "UNDERPRODUCTION", "underproduction"

    return "OVERPRODUCTION", "overproduction"


def candidate_levels(column, current_value, integer=False):
    """
    Historical quantiles are candidate values, not confirmed
    factory capacity constraints.
    """
    values = numeric_series(column)

    if values.empty:
        return [current_value]

    quantiles = values.quantile(
        [0.10, 0.30, 0.50, 0.70, 0.90]
    ).tolist()

    levels = []

    for value in quantiles + [current_value]:
        value = int(round(value)) if integer else round(float(value), 2)

        if value > 0 and value not in levels:
            levels.append(value)

    return sorted(levels)


def find_scenarios(context, planned, tolerance):
    worker_levels = candidate_levels(
        "no_of_workers", context["workers"], integer=True
    )
    machine_levels = candidate_levels(
        "machine_count", context["machines"], integer=True
    )
    hour_levels = candidate_levels(
        "working_hours", context["hours"]
    )

    rows = []

    for workers, hours, machines in product(
        worker_levels, hour_levels, machine_levels
    ):
        # Do not evaluate values outside the configured limits.
        if not WORKER_MIN <= workers <= WORKER_MAX:
            continue
        if not MACHINE_MIN <= machines <= MACHINE_MAX:
            continue
        if not HOUR_MIN <= hours <= HOUR_MAX:
            continue

        candidate_input = make_input(
            context["month"],
            context["week"],
            context["week_month"],
            context["factory"],
            context["family"],
            context["item"],
            context["size"],
            context["grade"],
            context["rate"],
            workers,
            hours,
            machines,
            context["bottleneck"],
            context["category"]
        )

        try:
            predicted = predict_quantity(candidate_input)
        except Exception:
            continue

        gap = predicted - planned

        rows.append({
            "Workers": workers,
            "Total weekly labour-hours": hours,
            "Machines": machines,
            "Predicted quantity": round(predicted, 2),
            "Difference from plan": round(gap, 2),
            "Absolute gap": abs(gap),
            "Within tolerance": abs(gap) <= tolerance,
            # Temporary proxy, not a real financial cost.
            "Resource-use proxy": workers + hours + machines
        })

    scenarios = pd.DataFrame(rows)

    if scenarios.empty:
        return scenarios

    return scenarios.sort_values(
        ["Within tolerance", "Absolute gap", "Resource-use proxy"],
        ascending=[False, True, True]
    ).reset_index(drop=True)


def display_status(title, detail, status_type):
    st.markdown(
        f"""
        <div class="status-card status-{status_type}">
            <div class="status-title">{title}</div>
            <div class="status-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# 6. SIDEBAR: MODEL AND INPUT BOUNDARIES
# ==========================================================

with st.sidebar:
    st.header("⚙️ Model information")
    st.write("**Prediction target:** `actual_qty`")
    st.write("**Input features:** 16")
    st.write(f"**Historical records:** {len(df):,}")

    st.divider()

    st.subheader("Input boundaries")
    st.caption("Provisional historical minimum and maximum")

    st.write(f"Workers: **{WORKER_MIN:g}–{WORKER_MAX:g}**")
    st.write(f"Machines: **{MACHINE_MIN:g}–{MACHINE_MAX:g}**")
    st.write(f"Weekly labour-hours: **{HOUR_MIN:.2f}–{HOUR_MAX:.2f}**")
    st.write(f"Rate: **{RATE_MIN:.2f}–{RATE_MAX:.2f}**")

    st.warning(
        "These ranges come from historical data. Replace them with "
        "factory-approved operating limits before operational use."
    )


# ==========================================================
# 7. PRODUCT AND FACTORY CONTEXT
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("1. Select the production context")

col1, col2 = st.columns(2)

with col1:
    factory = st.selectbox(
        "Factory unit",
        get_options(df["factory_unit"])
    )

factory_df = df[df["factory_unit"] == factory]

with col2:
    family = st.selectbox(
        "Product family",
        get_options(factory_df["product_family"])
    )

family_df = factory_df[factory_df["product_family"] == family]

item = st.selectbox(
    "Item",
    get_options(family_df["item_name"])
)

item_df = family_df[family_df["item_name"] == item]

size = st.selectbox(
    "Size",
    get_options(item_df["size_id"])
)

size_df = item_df[item_df["size_id"] == size]

grade = st.selectbox(
    "Grade",
    get_options(size_df["grade"])
)

grade_df = size_df[size_df["grade"] == grade]

category = st.selectbox(
    "Order category",
    get_options(grade_df["order_category"])
)

context_df = grade_df[grade_df["order_category"] == category]

if context_df.empty:
    st.error("No matching historical context was found.")
    st.stop()

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# 8. RESOURCE INPUTS AND PLANNED QUANTITY
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("2. Enter weekly production conditions")

st.info(
    "One record represents one product for one week. "
    "Working hours means the total labour-hours across all workers "
    "during that week."
)

month_options = get_options(df["financial_month"])
month_mode = context_df["financial_month"].mode()

default_month = (
    month_mode.iloc[0] if not month_mode.empty else month_options[0]
)

default_workers = int(np.clip(
    round(median_value(context_df["no_of_workers"], 16)),
    WORKER_MIN, WORKER_MAX
))
default_machines = int(np.clip(
    round(median_value(context_df["machine_count"], 4)),
    MACHINE_MIN, MACHINE_MAX
))
default_hours = float(np.clip(
    median_value(context_df["working_hours"], 600),
    HOUR_MIN, HOUR_MAX
))
default_rate = float(np.clip(
    median_value(context_df["rate"], RATE_MIN),
    RATE_MIN, RATE_MAX
))
default_week = int(np.clip(
    round(median_value(context_df["week_no"], 1)), 1, 53
))
default_week_month = int(np.clip(
    round(median_value(context_df["week_of_month"], 1)), 1, 5
))

a, b, c = st.columns(3)

with a:
    month = st.selectbox(
        "Financial month",
        month_options,
        index=month_options.index(default_month)
    )

    week = st.number_input(
        "Week number",
        min_value=1,
        max_value=53,
        value=default_week,
        step=1
    )

    week_month = st.number_input(
        "Week of month",
        min_value=1,
        max_value=5,
        value=default_week_month,
        step=1
    )

with b:
    rate = st.number_input(
        "Rate",
        min_value=float(RATE_MIN),
        max_value=float(RATE_MAX),
        value=default_rate,
        step=1.0,
        format="%.2f",
        help="Allowed range is based on the historical dataset."
    )

    workers = st.number_input(
        "Number of workers",
        min_value=int(np.ceil(WORKER_MIN)),
        max_value=int(np.floor(WORKER_MAX)),
        value=default_workers,
        step=1
    )

    hours = st.number_input(
        "Total labour-hours for the week",
        min_value=float(HOUR_MIN),
        max_value=float(HOUR_MAX),
        value=default_hours,
        step=1.0,
        format="%.2f"
    )

with c:
    machines = st.number_input(
        "Number of machines",
        min_value=int(np.ceil(MACHINE_MIN)),
        max_value=int(np.floor(MACHINE_MAX)),
        value=default_machines,
        step=1
    )

    bottleneck = st.selectbox(
        "Bottleneck present?",
        [False, True],
        format_func=lambda value: "Yes" if value else "No"
    )

    planned_qty = st.number_input(
        "Planned production quantity",
        min_value=0.01,
        value=35.0,
        step=1.0,
        format="%.2f",
        help="Enter the real planned quantity for this product and week."
    )

tolerance = st.number_input(
    "Acceptable difference from plan",
    min_value=0.0,
    value=1.0,
    step=0.5,
    help="Confirm the acceptable production tolerance with the factory."
)

st.caption(
    "Worker, machine, labour-hour, and rate inputs are bounded by "
    "observed historical values. The planned quantity is not restricted "
    "to the historical actual_qty range because a production target may "
    "legitimately differ from past output."
)

st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# 9. VALIDATION AND PREDICTION
# ==========================================================

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.header("3. Predict weekly actual production")

predict_clicked = st.button(
    "🚀 Predict production and generate recommendations",
    type="primary",
    use_container_width=True
)

if predict_clicked:
    errors = []

    errors += check_range(
        "Workers", workers, WORKER_MIN, WORKER_MAX
    )
    errors += check_range(
        "Machines", machines, MACHINE_MIN, MACHINE_MAX
    )
    errors += check_range(
        "Total weekly labour-hours", hours, HOUR_MIN, HOUR_MAX
    )
    errors += check_range(
        "Rate", rate, RATE_MIN, RATE_MAX
    )
    errors += check_range("Week number", week, WEEK_MIN, WEEK_MAX)
    errors += check_range(
        "Week of month", week_month,
        WEEK_MONTH_MIN, WEEK_MONTH_MAX
    )

    if not np.isfinite(float(planned_qty)) or planned_qty <= 0:
        errors.append("Planned quantity must be greater than zero.")

    if not np.isfinite(float(tolerance)) or tolerance < 0:
        errors.append("The acceptable difference cannot be negative.")

    if errors:
        st.error("INPUT VALIDATION FAILED — correct the following:")
        for error in errors:
            st.markdown(f"- 🔴 {error}")

        st.stop()

    current_input = make_input(
        month, week, week_month,
        factory, family, item, size, grade,
        rate, workers, hours, machines, bottleneck, category
    )

    try:
        prediction = predict_quantity(current_input)
    except Exception as exc:
        st.error("Prediction failed. Review the inputs and saved model.")
        st.code(str(exc))
        st.stop()

    gap = prediction - planned_qty
    status, status_key = production_status(
        prediction, planned_qty, tolerance
    )

    st.markdown("### Production summary")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Planned quantity", f"{planned_qty:,.2f}")
    metric2.metric("Predicted actual quantity", f"{prediction:,.2f}")
    metric3.metric("Difference from plan", f"{gap:+,.2f}")

    achievement = prediction / planned_qty * 100

    st.progress(min(achievement / 100, 1.0))
    st.write(f"**Predicted achievement:** {achievement:.2f}%")

    if status_key == "underproduction":
        display_status(
            "🔴 UNDERPRODUCTION",
            f"Predicted output is {abs(gap):,.2f} units below the plan. "
            "Review the improvement scenarios below.",
            "under"
        )

    elif status_key == "overproduction":
        display_status(
            "🟠 OVERPRODUCTION",
            f"Predicted output is {gap:,.2f} units above the plan. "
            "Review scenarios that reduce excess output.",
            "over"
        )

    else:
        display_status(
            "🟢 KEEP RESOURCES AS THEY ARE",
            "The prediction is within the selected tolerance. "
            "No changes are recommended.",
            "ok"
        )

    st.markdown('</div>', unsafe_allow_html=True)


    # ======================================================
    # 10. PRESCRIPTIVE RECOMMENDATIONS
    # ======================================================

    st.markdown(
        '<div class="prescriptive-card">',
        unsafe_allow_html=True
    )

    st.header("4. Prescriptive resource recommendations")

    context = {
        "month": month,
        "week": week,
        "week_month": week_month,
        "factory": factory,
        "family": family,
        "item": item,
        "size": size,
        "grade": grade,
        "rate": rate,
        "workers": workers,
        "hours": hours,
        "machines": machines,
        "bottleneck": bottleneck,
        "category": category
    }

    if status_key == "on_target":
        st.success(
            "RECOMMENDED ACTION: Keep the current resources as they are."
        )

        st.write("### Current resources")

        current_table = pd.DataFrame([{
            "Workers": int(workers),
            "Total weekly labour-hours": float(hours),
            "Machines": int(machines),
            "Predicted quantity": round(prediction, 2)
        }])

        st.dataframe(
            current_table,
            use_container_width=True,
            hide_index=True
        )

        st.write(
            "The current prediction is within the tolerance you selected. "
            "Avoid changing resources solely to improve the model output."
        )

        with st.spinner("Preparing optional comparison scenarios..."):
            scenarios = find_scenarios(context, planned_qty, tolerance)

        if not scenarios.empty:
            with st.expander("Optional: compare other resource combinations"):
                st.dataframe(
                    scenarios.head(10),
                    use_container_width=True,
                    hide_index=True
                )

    else:
        if status_key == "underproduction":
            st.markdown("### 🟣 Improvement plan")
            st.write(
                "The system searches for alternatives predicted to move "
                "production closer to the planned quantity. Review whether "
                "more workers, additional machines, or more total "
                "labour-hours are feasible."
            )
        else:
            st.markdown("### 🟦 Excess-production reduction plan")
            st.write(
                "The system searches for alternatives predicted to move "
                "production closer to the planned quantity. Review whether "
                "workers, machines, or total labour-hours can be reduced "
                "without violating operational requirements."
            )

        with st.spinner("Evaluating resource combinations..."):
            scenarios = find_scenarios(context, planned_qty, tolerance)

        if scenarios.empty:
            st.error(
                "No candidate scenarios could be evaluated. "
                "This does not prove that no feasible solution exists."
            )

        else:
            changed = scenarios[
                (scenarios["Workers"] != workers)
                | (scenarios["Total weekly labour-hours"] != hours)
                | (scenarios["Machines"] != machines)
            ].copy()

            target_scenarios = changed[
                changed["Within tolerance"]
            ]

            if status_key == "underproduction":
                if not target_scenarios.empty:
                    recommendations = target_scenarios.head(5)
                    reason = (
                        "These candidates are predicted to meet the target "
                        "within tolerance."
                    )
                else:
                    below_plan = changed[
                        changed["Predicted quantity"] <= planned_qty
                    ]

                    if not below_plan.empty:
                        recommendations = below_plan.sort_values(
                            "Predicted quantity",
                            ascending=False
                        ).head(5)
                        reason = (
                            "No tested candidate met tolerance. These are "
                            "the closest candidates below the plan."
                        )
                    else:
                        recommendations = changed.head(5)
                        reason = (
                            "No tested candidate met tolerance. Review "
                            "these alternatives carefully."
                        )

            else:
                if not target_scenarios.empty:
                    recommendations = target_scenarios.sort_values(
                        ["Resource-use proxy", "Absolute gap"],
                        ascending=[True, True]
                    ).head(5)
                    reason = (
                        "These candidates are within tolerance, prioritised "
                        "by a provisional resource-use proxy."
                    )
                else:
                    recommendations = changed.sort_values(
                        ["Absolute gap", "Resource-use proxy"],
                        ascending=[True, True]
                    ).head(5)
                    reason = (
                        "No tested candidate met tolerance. These options "
                        "are closest to the plan, with the provisional "
                        "resource-use proxy used as a secondary ranking."
                    )

            if recommendations.empty:
                st.warning(
                    "No different resource combination was found in the "
                    "candidate grid. The current resource combination "
                    "remains the only available option in this search."
                )

            else:
                st.write(reason)

                st.markdown("### Current resources")

                current_table = pd.DataFrame([{
                    "Workers": int(workers),
                    "Total weekly labour-hours": float(hours),
                    "Machines": int(machines),
                    "Predicted quantity": round(prediction, 2),
                    "Difference from plan": round(gap, 2)
                }])

                st.dataframe(
                    current_table,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("### Proposed alternatives")

                st.dataframe(
                    recommendations[
                        [
                            "Workers",
                            "Total weekly labour-hours",
                            "Machines",
                            "Predicted quantity",
                            "Difference from plan",
                            "Within tolerance"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                best = recommendations.iloc[0]

                st.markdown("### First candidate to review")

                delta1, delta2, delta3 = st.columns(3)

                delta1.metric(
                    "Workers",
                    f"{int(best['Workers'])}",
                    delta=f"{int(best['Workers']) - int(workers):+d}"
                )

                delta2.metric(
                    "Total weekly labour-hours",
                    f"{best['Total weekly labour-hours']:.2f}",
                    delta=(
                        f"{best['Total weekly labour-hours'] - hours:+.2f}"
                    )
                )

                delta3.metric(
                    "Machines",
                    f"{int(best['Machines'])}",
                    delta=f"{int(best['Machines']) - int(machines):+d}"
                )

                st.markdown("### What should change?")

                changes = []

                if int(best["Workers"]) != int(workers):
                    changes.append(
                        f"Workers: {int(workers)} → "
                        f"{int(best['Workers'])}"
                    )

                if not np.isclose(
                    best["Total weekly labour-hours"], hours
                ):
                    changes.append(
                        f"Total weekly labour-hours: {hours:.2f} → "
                        f"{best['Total weekly labour-hours']:.2f}"
                    )

                if int(best["Machines"]) != int(machines):
                    changes.append(
                        f"Machines: {int(machines)} → "
                        f"{int(best['Machines'])}"
                    )

                if changes:
                    for change in changes:
                        st.markdown(f"- **{change}**")
                else:
                    st.write("No resource changes are proposed.")

                st.write(
                    f"Candidate predicted quantity: "
                    f"**{best['Predicted quantity']:.2f}**"
                )

                st.write(
                    f"Candidate difference from plan: "
                    f"**{best['Difference from plan']:+.2f}**"
                )

                st.warning(
                    "This is a model-evaluated scenario, not a guaranteed "
                    "production outcome. Verify staffing, machine "
                    "availability, shift limits, quality requirements, "
                    "delivery deadlines, and costs with the factory."
                )

        st.download_button(
            "⬇️ Download evaluated scenarios",
            data=scenarios.to_csv(index=False).encode("utf-8")
            if not scenarios.empty else b"",
            file_name="di_leather_resource_scenarios.csv",
            mime="text/csv",
            disabled=scenarios.empty
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# 11. FOOTER
# ==========================================================

st.divider()

st.caption(
    "DI Leather Production Intelligence | "
    "Model predictions are estimates. Validate all proposed resource "
    "changes with the production team."
)




