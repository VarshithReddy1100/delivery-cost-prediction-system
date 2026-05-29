import os
import json
import requests
import joblib
import pandas as pd
import streamlit as st


API_URL = os.getenv(
    "BACKEND_API_URL",
    "http://127.0.0.1:8000/predict"
)
MODEL_PATH = "models/delivery_cost_model.pkl"
METRICS_PATH = "models/model_metrics.json"

st.set_page_config(
    page_title="ShipCost AI",
    page_icon="🚚",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: #0f172a;
}
.main-title {
    font-size: 40px;
    font-weight: 900;
    color: #0f172a;
}
.sub-title {
    font-size: 17px;
    color: #64748b;
}
.kpi-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    border-left: 5px solid #2563eb;
}
.prediction-card {
    background: linear-gradient(135deg, #2563eb, #0f172a);
    color: white;
    padding: 34px;
    border-radius: 24px;
    text-align: center;
    box-shadow: 0 8px 28px rgba(0,0,0,0.25);
}
.logo {
    font-size: 26px;
    font-weight: 900;
    color: white;
    margin-bottom: 20px;
}
.small-text {
    color: #64748b;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as file:
            return json.load(file)
    return None


def predict_cost(input_data):
    try:
        response = requests.post(API_URL, json=input_data, timeout=4)
        if response.status_code == 200:
            return response.json()["estimated_delivery_cost"], "FastAPI Backend"
    except Exception:
        pass

    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return round(float(prediction), 2), "Local Model Fallback"


def cost_breakdown(cost):
    return pd.DataFrame({
        "Component": [
            "Base Fee",
            "Distance Fee",
            "Weight Handling",
            "Priority / Service Fee"
        ],
        "Amount (₹)": [
            round(cost * 0.30, 2),
            round(cost * 0.35, 2),
            round(cost * 0.20, 2),
            round(cost * 0.15, 2)
        ]
    })


if "history" not in st.session_state:
    st.session_state.history = []


with st.sidebar:
    st.markdown('<div class="logo">🚚 ShipCost AI</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🤖 Cost Estimator",
            "📊 Analytics",
            "📈 Model Metrics",
            "🧪 API Payload",
            "⚙️ Settings"
        ]
    )

    st.divider()

    st.markdown("### 📦 Shipment Inputs")

    delivery_partner = st.selectbox(
        "Delivery Partner",
        ["delhivery", "xpressbees", "shadowfax", "dhl"]
    )

    package_type = st.selectbox(
        "Package Type",
        ["electronics", "groceries", "clothing", "cosmetics", "automobile parts"]
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["bike", "van", "truck", "ev van"]
    )

    delivery_mode = st.selectbox(
        "Delivery Mode",
        ["standard", "express", "same day"]
    )

    region = st.selectbox(
        "Region",
        ["north", "south", "east", "west", "central"]
    )

    weather_condition = st.selectbox(
        "Weather",
        ["clear", "rainy", "foggy", "stormy", "windy"]
    )

    distance_km = st.slider("Distance (km)", 1.0, 500.0, 25.5)
    package_weight_kg = st.slider("Weight (kg)", 0.1, 100.0, 8.2)
    delivery_time_hours = st.slider("Delivery Time (hrs)", 0.1, 72.0, 5.0)
    expected_time_hours = st.slider("Expected Time (hrs)", 0.1, 72.0, 4.5)

    delayed = st.radio("Delayed?", ["no", "yes"], horizontal=True)

    delivery_status = st.selectbox(
        "Delivery Status",
        ["delivered", "in transit", "cancelled", "returned"]
    )

    delivery_rating = st.slider("Rating", 1, 5, 4)


input_data = {
    "delivery_partner": delivery_partner,
    "package_type": package_type,
    "vehicle_type": vehicle_type,
    "delivery_mode": delivery_mode,
    "region": region,
    "weather_condition": weather_condition,
    "distance_km": distance_km,
    "package_weight_kg": package_weight_kg,
    "delivery_time_hours": delivery_time_hours,
    "expected_time_hours": expected_time_hours,
    "delayed": delayed,
    "delivery_status": delivery_status,
    "delivery_rating": delivery_rating
}


st.markdown(
    '<div class="main-title">Real-Time Delivery Cost Estimation System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">AI-powered logistics pricing using Machine Learning, FastAPI, and Streamlit</div>',
    unsafe_allow_html=True
)

st.divider()


if page == "🏠 Overview":
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Distance", f"{distance_km} km")
    c2.metric("Weight", f"{package_weight_kg} kg")
    c3.metric("Vehicle", vehicle_type.title())
    c4.metric("Mode", delivery_mode.title())

    st.markdown("### 🚀 Project Summary")

    st.info(
        "This dashboard predicts delivery cost using a trained ML regression model. "
        "It supports FastAPI prediction and local model fallback if the API is offline."
    )

    st.markdown("### 🧠 ML Workflow")

    workflow = pd.DataFrame({
        "Phase": [
            "Data Loading",
            "EDA",
            "Preprocessing",
            "Training",
            "Evaluation",
            "Saving",
            "Deployment"
        ],
        "Description": [
            "Load Kaggle logistics dataset",
            "Analyze distributions, nulls, and correlations",
            "Impute, scale, and one-hot encode features",
            "Train Linear Regression, Random Forest, Gradient Boosting",
            "Compare MAE, RMSE, and R² Score",
            "Save best model using Joblib",
            "Serve using FastAPI and Streamlit"
        ]
    })

    st.dataframe(workflow, use_container_width=True, hide_index=True)


elif page == "🤖 Cost Estimator":
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### 📍 Shipment Details")

        pickup = st.text_input("Pickup Location", "Hyderabad")
        drop = st.text_input("Drop-off Location", "Bangalore")

        col_a, col_b, col_c = st.columns(3)
        length = col_a.number_input("Length (cm)", 1.0, 300.0, 30.0)
        width = col_b.number_input("Width (cm)", 1.0, 300.0, 20.0)
        height = col_c.number_input("Height (cm)", 1.0, 300.0, 15.0)

        goods_value = st.number_input(
            "Value of Goods (₹)",
            min_value=0.0,
            max_value=1000000.0,
            value=5000.0
        )

        volume = length * width * height

        st.markdown("### 📋 Current ML Features")
        st.json(input_data)

    with right:
        st.markdown("### 💰 Estimate Panel")

        if st.button("🚀 Get Delivery Cost Estimate", use_container_width=True):
            with st.spinner("Running ML prediction..."):
                cost, source = predict_cost(input_data)

            st.markdown(
                f"""
                <div class="prediction-card">
                    <h3>Estimated Delivery Cost</h3>
                    <h1>₹ {cost}</h1>
                    <p>Source: {source}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 💵 Cost Breakdown")
            st.dataframe(
                cost_breakdown(cost),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### ⏱ Estimated Delivery Window")
            st.success(f"{expected_time_hours} to {delivery_time_hours} hours")

            confidence = "High" if delivery_rating >= 4 and delayed == "no" else "Medium"
            st.markdown("### 🎯 Confidence")
            st.info(f"{confidence} confidence estimate")

            st.session_state.history.append({
                "Pickup": pickup,
                "Drop": drop,
                "Distance": distance_km,
                "Weight": package_weight_kg,
                "Cost": cost,
                "Source": source
            })

        st.caption(
            "Disclaimer: This is an estimated delivery cost. Actual charges may vary due to weather, traffic, route, and operational conditions."
        )


elif page == "📊 Analytics":
    st.markdown("### 📊 Feature Analytics")

    chart_df = pd.DataFrame({
        "Feature": [
            "Distance",
            "Weight",
            "Delivery Time",
            "Expected Time",
            "Rating"
        ],
        "Value": [
            distance_km,
            package_weight_kg,
            delivery_time_hours,
            expected_time_hours,
            delivery_rating
        ]
    })

    st.bar_chart(chart_df.set_index("Feature"))

    st.markdown("### 🕘 Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.line_chart(history_df["Cost"])
    else:
        st.warning("No predictions made yet.")


elif page == "📈 Model Metrics":
    st.markdown("### 📈 Model Performance")

    metrics = load_metrics()

    if metrics:
        st.success(f"Best Model: {metrics['best_model']}")
        st.info(f"Best R² Score: {metrics['best_r2_score']}")

        metric_df = pd.DataFrame(metrics["all_model_metrics"]).T
        st.dataframe(metric_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### RMSE Comparison")
            st.bar_chart(metric_df["RMSE"])

        with col2:
            st.markdown("### R² Comparison")
            st.bar_chart(metric_df["R2_Score"])
    else:
        st.error("Run training first. model_metrics.json not found.")


elif page == "🧪 API Payload":
    st.markdown("### 🧪 Current API Payload")
    st.json(input_data)

    st.markdown("### Endpoint")
    st.code("POST http://127.0.0.1:8000/predict")

    st.info(
        "Only trained ML features are sent to the API. Extra UI fields such as pickup, drop, dimensions, and goods value are kept for professional user experience and future model improvement."
    )


elif page == "⚙️ Settings":
    st.markdown("### ⚙️ System Configuration")

    st.write("FastAPI URL")
    st.code(API_URL)

    st.write("Model Path")
    st.code(MODEL_PATH)

    st.write("Metrics Path")
    st.code(METRICS_PATH)

    st.markdown("### Deployment Note")
    st.warning(
        "Run FastAPI first using: uvicorn backend.app:app --reload. "
        "If FastAPI is down, the dashboard automatically uses local model fallback."
    )


st.divider()
st.caption("Built with Python • Scikit-Learn • FastAPI • Streamlit • Machine Learning")