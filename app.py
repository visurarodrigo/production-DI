
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DI Leather | Production Intelligence",
    page_icon="👞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATHS AND MODEL FEATURES
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "production_model.joblib"
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_features.csv"
IMAGE_PATH = PROJECT_DIR / "assets" / "di_leather_shoes.png"

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
    "avg_hours_per_worker",
]

TARGET = "actual_qty"

# These are provisional historical limits, not factory-approved limits.
RESOURCE_COLUMNS = [
    "no_of_workers",
    "working_hours",
    "machine_count",
    "rate",
]


# ============================================================
# 3. COMPANY DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #071426 0%, #0b1d33 55%, #102743 100%);
        color: #edf4ff;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06101f 0%, #102641 100%);
        border-right: 1px solid #25466a;
    }

    [data-testid="stSidebar"] * {
        color: #e8f1ff;
    }

    .hero {
        background: linear-gradient(120deg, #102845, #173b60);
        border: 1px solid #315b82;
        border-radius: 22px;
        padding: 30px;
        margin-bottom: 22px;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.20);
    }

    .eyebrow {
        color: #7de3f5;
        font-weight: 800;
        letter-spacing: 2px;
        font-size: 0.8rem;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.5rem;
        line-height: 1.15;
        margin: 12px 0;
    }

    .hero p {
        color: #c6d8ed;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-card {
        background: #10243b;
        border: 1px solid #294767;
        border-radius: 17px;
        padding: 20px;
        margin: 8px 0 18px 0;
    }

    .status-card {
        border-radius: 18px;
        padding: 25px;
        margin: 14px 0 20px 0;
        border: 1px solid rgba(255,255,255,0.15);
    }

    .status-card h2 {
        color: white;
        font-size: 2rem;
        margin: 5px 0 10px 0;
    }

    .status-card p {
        color: #f0f5ff;
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
    }

    .status-under {
        background: linear-gradient(120deg, #651e2c, #9f293a);
        border-color: #ff6474;
    }

    .status-over {
        background: linear-gradient(120deg, #633617, #a95720);
        border-color: #ffad62;
    }

    .status-keep {
        background: linear-gradient(120deg, #123e35, #17634e);
        border-color: #58d6a5;
    }

    .prescriptive {
        background: linear-gradient(145deg, #211c49, #173653);
        border: 1px solid #7666d8;
        border-radius: 20px;
        padding: 24px;
        margin-top: 20px;
    }

    .prescriptive h2, .prescriptive h3 {
        color: #ffffff;
    }

    .resource-current {
        background: #142c45;
        border: 1px solid #3b6285;
        border-radius: 14px;
        padding: 18px;
    }

    .resource-proposed {
        background: #202044;
        border: 1px solid #8b7cf6;
        border-radius: 14px;
        padding: 18px;
    }

    .small-label {
        font-size: 0.85rem;
        color: #b9cde3;
        margin-bottom: 5px;
    }

    div[data-testid="stMetric"] {
        background: #10243b;
        border: 1px solid #294767;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="stMetric"] label {
        color: #b9cde3;
    }

    .footer-note {
        color: #9db3cc;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 20px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2575d8, #4b67e8);
        border: none;
        color: white;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #3989f0, #687dff);
        border: none;
        color: white;
    }

    h1, h2, h3 {
        color: #f4f8ff;
    }

    [data-testid="stCaptionContainer"] {
        color: #a9bed6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD MODEL AND HISTORICAL DATA
# ============================================================

@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


if not MODEL_PATH.exists():
    st.error(
        f"Could not find the production model at: {MODEL_PATH}"
    )
    st.info(
        "Check that models/production_model.joblib exists "
        "in the project folder."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        f"Could not find the processed dataset at: {DATA_PATH}"
    )
    st.info(
        "Check that data/processed/final_features.csv exists."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    historical_data = load_data(str(DATA_PATH))
except Exception as error:
    st.error("The model or historical dataset could not be loaded.")
    st.code(str(error))
    st.stop()

missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in historical_data.columns
]

if missing_columns:
    st.error(
        "The historical dataset is missing required columns: "
        + ", ".join(missing_columns)
    )
    st.stop()

if len(historical_data) == 0:
    st.error("The historical dataset contains no records.")
    st.stop()

historical_data = historical_data.dropna(
    subset=FEATURES + [TARGET]
).copy()

if historical_data.empty:
    st.error(
        "No complete historical records are available for the app."
    )
    st.stop()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def clean_options(series):
    """Return unique, non-empty historical values."""
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def historical_bounds(column):
    """Return finite historical minimum and maximum."""
    values = pd.to_numeric(
        historical_data[column], errors="coerce"
    )
    values = values[np.isfinite(values)]

    if values.empty:
        raise ValueError(
            f"No valid historical values for {column}."
        )

    return float(values.min()), float(values.max())


def predict_quantity(input_row):
    """Predict weekly actual production quantity."""
    ordered_input = input_row[FEATURES].copy()

    prediction = float(model.predict(ordered_input)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return max(0.0, prediction)


def get_status(predicted, target, tolerance):
    """Classify the predicted quantity relative to the target."""
    difference = predicted - target

    if abs(difference) <= tolerance:
        return "KEEP", "Keep resources unchanged"

    if difference < -tolerance:
        return "UNDER", "Underproduction"

    return "OVER", "Overproduction"


def get_resource_levels(column, current_value):
    """
    Create candidate resource values using historical quantiles.

    These values are a scenario-search grid, not confirmed
    factory operating limits.
    """
    minimum, maximum = historical_bounds(column)

    series = pd.to_numeric(
        historical_data[column], errors="coerce"
    ).dropna()

    quantiles = series.quantile(
        [0.10, 0.30, 0.50, 0.70, 0.90]
    ).tolist()

    values = [minimum, *quantiles, maximum, current_value]

    cleaned = []

    for value in values:
        value = float(value)

        if column in ["no_of_workers", "machine_count"]:
            value = int(round(value))
        else:
            value = round(value, 2)

        if minimum <= value <= maximum and value > 0:
            cleaned.append(value)

    return sorted(set(cleaned))


def build_candidate_input(
    base_input,
    workers,
    hours,
    machines,
):
    """Change resource values while retaining product context."""
    candidate = base_input.copy()

    candidate.loc[:, "no_of_workers"] = int(workers)
    candidate.loc[:, "working_hours"] = float(hours)
    candidate.loc[:, "machine_count"] = int(machines)

    # Recalculate engineered features consistently.
    candidate.loc[:, "machine_worker_ratio"] = (
        float(machines) / float(workers)
    )

    candidate.loc[:, "avg_hours_per_worker"] = (
        float(hours) / float(workers)
    )

    return candidate[FEATURES]


def make_recommendations(
    baseline_input,
    current_prediction,
    planned_quantity,
    tolerance,
):
    """
    Search alternative resource combinations.

    The objective is to find a prediction near the planned quantity.
    Resource use is a secondary tie-breaker only; it is not a
    monetary cost model.
    """
    current_workers = int(
        baseline_input.iloc[0]["no_of_workers"]
    )
    current_hours = float(
        baseline_input.iloc[0]["working_hours"]
    )
    current_machines = int(
        baseline_input.iloc[0]["machine_count"]
    )

    worker_levels = get_resource_levels(
        "no_of_workers", current_workers
    )
    hour_levels = get_resource_levels(
        "working_hours", current_hours
    )
    machine_levels = get_resource_levels(
        "machine_count", current_machines
    )

    rows = []

    for workers, hours, machines in product(
        worker_levels, hour_levels, machine_levels
    ):
        candidate = build_candidate_input(
            baseline_input,
            workers,
            hours,
            machines,
        )

        prediction = predict_quantity(candidate)
        gap = prediction - planned_quantity

        rows.append(
            {
                "no_of_workers": int(workers),
                "working_hours": float(hours),
                "machine_count": int(machines),
                "predicted_actual_qty": prediction,
                "gap": gap,
                "absolute_gap": abs(gap),
                "within_tolerance": abs(gap) <= tolerance,
                # A temporary numerical tie-breaker only.
                "resource_use_proxy": (
                    float(workers)
                    + float(hours)
                    + float(machines)
                ),
            }
        )

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        return candidates, None

    # Include the current resource scenario in the comparison.
    current_row = pd.DataFrame(
        [
            {
                "no_of_workers": current_workers,
                "working_hours": current_hours,
                "machine_count": current_machines,
                "predicted_actual_qty": current_prediction,
                "gap": current_prediction - planned_quantity,
                "absolute_gap": abs(
                    current_prediction - planned_quantity
                ),
                "within_tolerance": abs(
                    current_prediction - planned_quantity
                ) <= tolerance,
                "resource_use_proxy": (
                    current_workers
                    + current_hours
                    + current_machines
                ),
            }
        ]
    )

    candidates = pd.concat(
        [candidates, current_row], ignore_index=True
    )

    candidates = candidates.drop_duplicates(
        subset=[
            "no_of_workers",
            "working_hours",
            "machine_count",
        ]
    )

    candidates = candidates.sort_values(
        by=[
            "within_tolerance",
            "absolute_gap",
            "resource_use_proxy",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    best = candidates.iloc[0].to_dict()

    return candidates, best


# ============================================================
# 6. COMPANY SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 20px 0;">
            <div style="font-size:2.8rem;">👞</div>
            <h2 style="margin:0;">DI LEATHER</h2>
            <p style="color:#93b9e2; font-size:0.82rem;
                      letter-spacing:1.5px;">
                PRODUCTION INTELLIGENCE
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Production workspace")
    st.markdown("🏭  Weekly production planning")
    st.markdown("📈  Production prediction")
    st.markdown("🎯  Resource recommendations")

    st.divider()

    st.markdown(
        """
        <div style="
            background:linear-gradient(145deg,#132b49,#172b45);
            border:1px solid #294d70;
            border-radius:14px;
            padding:16px;
        ">
            <div style="font-size:1.1rem;font-weight:700;">
                Plan smarter.
            </div>
            <div style="font-size:0.9rem;color:#b6cbe3;
                        margin-top:6px;line-height:1.5;">
                Explore weekly production scenarios and
                resource adjustments using historical data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. BRANDED HERO HEADER
# ============================================================

hero_left, hero_right = st.columns(
    [1.15, 1],
    gap="large",
    vertical_alignment="center",
)

with hero_left:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                DI Leather | Operations
            </div>
            <h1>
                Production planning,<br>
                made smarter.
            </h1>
            <p>
                Predict weekly actual production, compare it with
                your planned quantity, and explore practical
                resource alternatives for each product.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with hero_right:
    if IMAGE_PATH.exists():
        st.image(
            str(IMAGE_PATH),
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="section-card"
                 style="text-align:center;padding:48px 20px;">
                <div style="font-size:4rem;">👞</div>
                <h3>DI Leather Production</h3>
                <p style="color:#b6cbe3;">
                    Weekly production planning and decision support
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Optional: place your leather-shoe image at "
            "assets/di_leather_shoes.png."
        )

st.markdown("## Weekly production workspace")
st.caption(
    "Select the product and enter the planned quantity and "
    "weekly operating conditions."
)


# ============================================================
# 8. INPUT BOUNDARIES
# ============================================================

try:
    worker_min, worker_max = historical_bounds("no_of_workers")
    hour_min, hour_max = historical_bounds("working_hours")
    machine_min, machine_max = historical_bounds("machine_count")
    rate_min, rate_max = historical_bounds("rate")
except Exception as error:
    st.error(str(error))
    st.stop()

worker_min = int(np.ceil(worker_min))
worker_max = int(np.floor(worker_max))
machine_min = int(np.ceil(machine_min))
machine_max = int(np.floor(machine_max))

# Remove any impossible boundary caused by malformed historical data.
if worker_min < 1 or worker_max < worker_min:
    st.error("The historical worker limits are invalid.")
    st.stop()

if machine_min < 1 or machine_max < machine_min:
    st.error("The historical machine limits are invalid.")
    st.stop()

if hour_min <= 0 or hour_max < hour_min:
    st.error("The historical labour-hour limits are invalid.")
    st.stop()

if rate_min <= 0 or rate_max < rate_min:
    st.error("The historical rate limits are invalid.")
    st.stop()


# ============================================================
# 9. PRODUCTION CONTEXT INPUTS
# ============================================================

st.markdown("### 1. Select the production context")

with st.container(border=True):
    context_col1, context_col2, context_col3 = st.columns(3)

    factory_options = clean_options(
        historical_data["factory_unit"]
    )
    family_options = clean_options(
        historical_data["product_family"]
    )
    item_options = clean_options(
        historical_data["item_name"]
    )
    size_options = clean_options(
        historical_data["size_id"]
    )
    grade_options = clean_options(
        historical_data["grade"]
    )
    category_options = clean_options(
        historical_data["order_category"]
    )
    month_options = clean_options(
        historical_data["financial_month"]
    )

    if not all(
        [
            factory_options,
            family_options,
            item_options,
            size_options,
            grade_options,
            category_options,
            month_options,
        ]
    ):
        st.error(
            "Some product selection fields have no valid "
            "historical options."
        )
        st.stop()

    with context_col1:
        factory_unit = st.selectbox(
            "Factory unit",
            factory_options,
        )

        product_family = st.selectbox(
            "Product family",
            family_options,
        )

        financial_month = st.selectbox(
            "Financial month",
            month_options,
        )

    with context_col2:
        # Product selection is narrowed by family and factory.
        matching_items = historical_data[
            (historical_data["factory_unit"] == factory_unit)
            & (
                historical_data["product_family"]
                == product_family
            )
        ]

        available_items = clean_options(
            matching_items["item_name"]
        )

        if not available_items:
            available_items = item_options

        item_name = st.selectbox(
            "Item",
            available_items,
        )

        item_rows = historical_data[
            (historical_data["factory_unit"] == factory_unit)
            & (
                historical_data["product_family"]
                == product_family
            )
            & (historical_data["item_name"] == item_name)
        ]

        if item_rows.empty:
            item_rows = historical_data[
                historical_data["item_name"] == item_name
            ]

        available_sizes = clean_options(item_rows["size_id"])
        available_grades = clean_options(item_rows["grade"])

        if not available_sizes:
            available_sizes = size_options
        if not available_grades:
            available_grades = grade_options

        size_id = st.selectbox(
            "Size",
            available_sizes,
        )

        grade = st.selectbox(
            "Grade",
            available_grades,
        )

    with context_col3:
        order_category = st.selectbox(
            "Order category",
            category_options,
        )

        week_no = st.number_input(
            "Week number",
            min_value=1,
            max_value=53,
            value=22,
            step=1,
            help="Enter the calendar or production week number.",
        )

        week_of_month = st.number_input(
            "Week of month",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )


# ============================================================
# 10. PLANNED QUANTITY AND RESOURCE INPUTS
# ============================================================

st.markdown("### 2. Enter the weekly production plan")

with st.container(border=True):
    target_col1, target_col2 = st.columns(2)

    with target_col1:
        planned_quantity = st.number_input(
            "Planned weekly production quantity",
            min_value=0.01,
            max_value=1_000_000_000.0,
            value=50.0,
            step=1.0,
            format="%.2f",
            help=(
                "Enter the real target for this product and week. "
                "Use the same unit as actual_qty in the historical data."
            ),
        )

    with target_col2:
        tolerance = st.number_input(
            "Acceptable difference from target",
            min_value=0.0,
            max_value=1_000_000_000.0,
            value=2.0,
            step=0.5,
            format="%.2f",
            help=(
                "If the prediction is within this many units "
                "of the target, the recommendation is to keep "
                "the current resources."
            ),
        )

st.markdown("### 3. Enter the current weekly resources")

st.caption(
    "Inputs are checked against historical data ranges. "
    "These ranges are provisional until confirmed by DI Leather."
)

resource_col1, resource_col2, resource_col3 = st.columns(3)

with resource_col1:
    no_of_workers = st.number_input(
        f"Number of workers ({worker_min}–{worker_max})",
        min_value=worker_min,
        max_value=worker_max,
        value=int(
            np.clip(
                round(
                    pd.to_numeric(
                        historical_data["no_of_workers"],
                        errors="coerce",
                    ).median()
                ),
                worker_min,
                worker_max,
            )
        ),
        step=1,
    )

    machine_count = st.number_input(
        f"Number of machines ({machine_min}–{machine_max})",
        min_value=machine_min,
        max_value=machine_max,
        value=int(
            np.clip(
                round(
                    pd.to_numeric(
                        historical_data["machine_count"],
                        errors="coerce",
                    ).median()
                ),
                machine_min,
                machine_max,
            )
        ),
        step=1,
    )

with resource_col2:
    working_hours = st.number_input(
        f"Total weekly labour-hours ({hour_min:.2f}–{hour_max:.2f})",
        min_value=float(hour_min),
        max_value=float(hour_max),
        value=float(
            np.clip(
                pd.to_numeric(
                    historical_data["working_hours"],
                    errors="coerce",
                ).median(),
                hour_min,
                hour_max,
            )
        ),
        step=1.0,
        format="%.2f",
        help=(
            "This is the total labour-hours across all workers "
            "for the week, not the hours for each worker."
        ),
    )

    rate = st.number_input(
        f"Rate ({rate_min:.2f}–{rate_max:.2f})",
        min_value=float(rate_min),
        max_value=float(rate_max),
        value=float(
            np.clip(
                pd.to_numeric(
                    historical_data["rate"],
                    errors="coerce",
                ).median(),
                rate_min,
                rate_max,
            )
        ),
        step=10.0,
        format="%.2f",
    )

with resource_col3:
    bottleneck_flag = st.selectbox(
        "Is a bottleneck present?",
        options=[False, True],
        format_func=lambda value: (
            "Yes" if value else "No"
        ),
    )

    st.markdown(
        """
        <div class="section-card">
            <div style="font-weight:700;color:#c8dafa;">
                Resource guidance
            </div>
            <p style="color:#a9bed6;line-height:1.6;">
                Use resources that are available for the selected
                week. The app checks historical boundaries but
                cannot confirm real-time factory availability.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 11. VALIDATION
# ============================================================

validation_errors = []

if not np.isfinite(float(planned_quantity)) or planned_quantity <= 0:
    validation_errors.append(
        "Planned production quantity must be greater than zero."
    )

if not np.isfinite(float(tolerance)) or tolerance < 0:
    validation_errors.append(
        "The acceptable difference must be zero or greater."
    )

if not worker_min <= int(no_of_workers) <= worker_max:
    validation_errors.append(
        f"Workers must be between {worker_min} and {worker_max}."
    )

if not machine_min <= int(machine_count) <= machine_max:
    validation_errors.append(
        f"Machines must be between {machine_min} and {machine_max}."
    )

if not hour_min <= float(working_hours) <= hour_max:
    validation_errors.append(
        f"Weekly labour-hours must be between "
        f"{hour_min:.2f} and {hour_max:.2f}."
    )

if not rate_min <= float(rate) <= rate_max:
    validation_errors.append(
        f"Rate must be between {rate_min:.2f} and {rate_max:.2f}."
    )

# A large mismatch may indicate a unit or target-entry error.
historical_target_median = float(
    pd.to_numeric(
        historical_data[TARGET], errors="coerce"
    ).median()
)

if (
    historical_target_median > 0
    and planned_quantity > historical_target_median * 10
):
    st.warning(
        "Your planned quantity is more than 10 times the "
        "historical median actual quantity. Check the unit and "
        "target carefully before interpreting the recommendations."
    )

if validation_errors:
    st.error("Please correct the following input errors:")

    for error in validation_errors:
        st.markdown(f"- 🔴 {error}")


# ============================================================
# 12. PREDICTION ACTION
# ============================================================

st.markdown("---")
st.markdown("### 4. Predict and assess production")

predict_button = st.button(
    "Predict production and generate recommendations",
    type="primary",
    use_container_width=True,
    disabled=bool(validation_errors),
)

if predict_button:
    baseline_input = pd.DataFrame(
        [
            {
                "financial_month": financial_month,
                "week_no": int(week_no),
                "week_of_month": int(week_of_month),
                "factory_unit": factory_unit,
                "product_family": product_family,
                "item_name": item_name,
                "size_id": size_id,
                "grade": grade,
                "rate": float(rate),
                "no_of_workers": int(no_of_workers),
                "working_hours": float(working_hours),
                "machine_count": int(machine_count),
                "bottleneck_flag": bool(bottleneck_flag),
                "order_category": order_category,
                "machine_worker_ratio": (
                    float(machine_count) / float(no_of_workers)
                ),
                "avg_hours_per_worker": (
                    float(working_hours) / float(no_of_workers)
                ),
            }
        ],
        columns=FEATURES,
    )

    try:
        with st.spinner(
            "Predicting weekly production and evaluating alternatives..."
        ):
            current_prediction = predict_quantity(baseline_input)

            current_gap = current_prediction - planned_quantity

            status_code, status_label = get_status(
                current_prediction,
                planned_quantity,
                tolerance,
            )

            candidates, best_candidate = make_recommendations(
                baseline_input,
                current_prediction,
                planned_quantity,
                tolerance,
            )

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            "Check the model features and input values."
        )
        st.exception(error)
        st.stop()

    st.session_state["di_result"] = {
        "baseline_input": baseline_input,
        "current_prediction": current_prediction,
        "planned_quantity": float(planned_quantity),
        "tolerance": float(tolerance),
        "current_gap": float(current_gap),
        "status_code": status_code,
        "status_label": status_label,
        "candidates": candidates,
        "best_candidate": best_candidate,
    }


# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

if "di_result" in st.session_state:
    result = st.session_state["di_result"]

    current_prediction = result["current_prediction"]
    planned_quantity = result["planned_quantity"]
    tolerance = result["tolerance"]
    current_gap = result["current_gap"]
    status_code = result["status_code"]
    status_label = result["status_label"]
    candidates = result["candidates"]
    best_candidate = result["best_candidate"]

    # Do not show stale results as if they belonged to new inputs.
    current_signature = (
        factory_unit,
        product_family,
        item_name,
        str(size_id),
        str(grade),
        order_category,
        financial_month,
        int(week_no),
        int(week_of_month),
        float(planned_quantity),
        float(tolerance),
        int(no_of_workers),
        float(working_hours),
        int(machine_count),
        float(rate),
        bool(bottleneck_flag),
    )

    result_input = result["baseline_input"].iloc[0]

    stored_signature = (
        result_input["factory_unit"],
        result_input["product_family"],
        result_input["item_name"],
        str(result_input["size_id"]),
        str(result_input["grade"]),
        result_input["order_category"],
        result_input["financial_month"],
        int(result_input["week_no"]),
        int(result_input["week_of_month"]),
        float(result["planned_quantity"]),
        float(result["tolerance"]),
        int(result_input["no_of_workers"]),
        float(result_input["working_hours"]),
        int(result_input["machine_count"]),
        float(result_input["rate"]),
        bool(result_input["bottleneck_flag"]),
    )

    if current_signature != stored_signature:
        st.info(
            "Your inputs have changed. Click the prediction button "
            "again to refresh the result and recommendations."
        )
    else:
        st.markdown("---")
        st.markdown("## Production decision")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Planned quantity",
                f"{planned_quantity:,.2f}",
            )

        with metric2:
            st.metric(
                "Predicted actual quantity",
                f"{current_prediction:,.2f}",
            )

        with metric3:
            st.metric(
                "Predicted gap",
                f"{current_gap:+,.2f}",
                help=(
                    "Predicted actual quantity minus planned quantity. "
                    "Negative means below target; positive means above."
                ),
            )

        if status_code == "UNDER":
            st.markdown(
                f"""
                <div class="status-card status-under">
                    <div class="eyebrow">Production status</div>
                    <h2>🔴 UNDERPRODUCTION</h2>
                    <p>
                        The model predicts approximately
                        <b>{abs(current_gap):,.2f}</b> units below
                        the planned quantity. Review the resource
                        alternatives below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif status_code == "OVER":
            st.markdown(
                f"""
                <div class="status-card status-over">
                    <div class="eyebrow">Production status</div>
                    <h2>🟠 OVERPRODUCTION</h2>
                    <p>
                        The model predicts approximately
                        <b>{abs(current_gap):,.2f}</b> units above
                        the planned quantity. Review whether resources
                        can be adjusted without compromising delivery
                        or quality.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                """
                <div class="status-card status-keep">
                    <div class="eyebrow">Production status</div>
                    <h2>🟢 KEEP RESOURCES UNCHANGED</h2>
                    <p>
                        The predicted quantity is within the
                        acceptable target difference. Your current
                        inputs are suitable under the selected
                        tolerance; no resource change is recommended.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ====================================================
        # 14. PRESCRIPTIVE RECOMMENDATIONS
        # ====================================================

        st.markdown(
            """
            <div class="prescriptive">
                <div class="eyebrow">Prescriptive decision support</div>
                <h2>Resource recommendations</h2>
                <p style="color:#c7d4ee;line-height:1.6;">
                    Compare the current resource allocation with
                    an alternative evaluated by the prediction model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_workers = int(no_of_workers)
        current_hours = float(working_hours)
        current_machines = int(machine_count)

        if status_code == "KEEP":
            st.success(
                "Recommendation: keep the current resources. "
                "The predicted quantity is already within tolerance."
            )

            st.markdown("### Current resources")

            keep1, keep2, keep3 = st.columns(3)

            keep1.metric("Workers", current_workers)
            keep2.metric("Weekly labour-hours", f"{current_hours:,.2f}")
            keep3.metric("Machines", current_machines)

            st.caption(
                "No resource change is required by this prototype. "
                "Continue to monitor real production and quality."
            )

        elif best_candidate is not None:
            proposed_workers = int(
                best_candidate["no_of_workers"]
            )
            proposed_hours = float(
                best_candidate["working_hours"]
            )
            proposed_machines = int(
                best_candidate["machine_count"]
            )
            proposed_prediction = float(
                best_candidate["predicted_actual_qty"]
            )
            proposed_gap = float(best_candidate["gap"])

            current_col, proposed_col = st.columns(2, gap="large")

            with current_col:
                st.markdown(
                    """
                    <div class="resource-current">
                        <h3>🔹 Current resources</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric("Workers", current_workers)
                st.metric(
                    "Weekly labour-hours",
                    f"{current_hours:,.2f}",
                )
                st.metric("Machines", current_machines)
                st.metric(
                    "Predicted quantity",
                    f"{current_prediction:,.2f}",
                )

            with proposed_col:
                st.markdown(
                    """
                    <div class="resource-proposed">
                        <h3>✨ Proposed resources</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric(
                    "Workers",
                    proposed_workers,
                    delta=proposed_workers - current_workers,
                )
                st.metric(
                    "Weekly labour-hours",
                    f"{proposed_hours:,.2f}",
                    delta=f"{proposed_hours - current_hours:+,.2f}",
                )
                st.metric(
                    "Machines",
                    proposed_machines,
                    delta=proposed_machines - current_machines,
                )
                st.metric(
                    "Predicted quantity",
                    f"{proposed_prediction:,.2f}",
                    delta=f"{proposed_prediction - current_prediction:+,.2f}",
                )

            st.markdown("### What should change?")

            changes = []

            if proposed_workers != current_workers:
                changes.append(
                    f"Workers: {current_workers} → {proposed_workers}"
                )

            if not np.isclose(
                proposed_hours, current_hours, atol=0.01
            ):
                changes.append(
                    f"Weekly labour-hours: {current_hours:,.2f} "
                    f"→ {proposed_hours:,.2f}"
                )

            if proposed_machines != current_machines:
                changes.append(
                    f"Machines: {current_machines} → {proposed_machines}"
                )

            if changes:
                for change in changes:
                    st.markdown(f"- {change}")
            else:
                st.info(
                    "The best scenario returned the same resource "
                    "allocation as the current scenario. The candidate "
                    "search did not identify a better alternative."
                )

            if abs(proposed_gap) <= tolerance:
                st.success(
                    "The proposed scenario is within the selected "
                    "acceptable difference from the target."
                )
            elif proposed_gap < -tolerance:
                st.warning(
                    f"The proposed scenario is still predicted to be "
                    f"{abs(proposed_gap):,.2f} units below target. "
                    "The tested historical resource combinations did "
                    "not identify a scenario within tolerance."
                )
            else:
                st.warning(
                    f"The proposed scenario is still predicted to be "
                    f"{proposed_gap:,.2f} units above target. "
                    "The tested historical resource combinations did "
                    "not identify a scenario within tolerance."
                )

            st.markdown("### Scenario comparison")

            comparison = pd.DataFrame(
                [
                    {
                        "Scenario": "Current resources",
                        "Workers": current_workers,
                        "Weekly labour-hours": current_hours,
                        "Machines": current_machines,
                        "Predicted quantity": current_prediction,
                        "Gap from target": current_gap,
                    },
                    {
                        "Scenario": "Proposed resources",
                        "Workers": proposed_workers,
                        "Weekly labour-hours": proposed_hours,
                        "Machines": proposed_machines,
                        "Predicted quantity": proposed_prediction,
                        "Gap from target": proposed_gap,
                    },
                ]
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View other tested resource scenarios"):
                display_candidates = candidates.copy()

                display_candidates = display_candidates.rename(
                    columns={
                        "no_of_workers": "Workers",
                        "working_hours": "Weekly labour-hours",
                        "machine_count": "Machines",
                        "predicted_actual_qty": "Predicted quantity",
                        "gap": "Gap from target",
                        "absolute_gap": "Absolute gap",
                        "within_tolerance": "Within tolerance",
                        "resource_use_proxy": "Temporary resource proxy",
                    }
                )

                st.dataframe(
                    display_candidates.head(20).round(2),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Scenarios are model estimates. The temporary "
                    "resource proxy is only a numerical tie-breaker; "
                    "it is not a monetary cost or a validated measure "
                    "of factory efficiency."
                )

        # ====================================================
        # 15. OPERATIONAL NOTE
        # ====================================================

        st.markdown("---")
        st.markdown("### Before implementing a recommendation")

        st.info(
            "Confirm worker availability, machine availability, "
            "shift limits, order deadlines, quality requirements, "
            "and actual operating costs with the production team. "
            "A model prediction is not a guaranteed production outcome."
        )

        # Download the evaluated scenarios for review.
        if best_candidate is not None:
            download_data = candidates.copy()

            st.download_button(
                label="Download evaluated resource scenarios",
                data=download_data.to_csv(index=False).encode("utf-8"),
                file_name="di_leather_resource_scenarios.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-note">
        DI Leather Production Intelligence · Weekly production
        planning and decision support
    </div>
    """,
    unsafe_allow_html=True,
)

