# Streamlit app for predictive maintenance (engine condition prediction)
import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FILENAME = "best_predictive_maintenance_model_v1.joblib"

# Feature order must match Xtrain.csv used during training
FEATURE_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "temp_differential",
]


@st.cache_resource
def load_model():
    # Load the pre-trained XGBoost pipeline committed by the pipeline (sits next to this file)
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    return joblib.load(model_path)


model = load_model()

st.title("Predictive Maintenance: Engine Condition Prediction")
st.write(
    """
This application predicts whether an engine requires maintenance based on live sensor readings.
Enter the sensor values below to get a prediction.
"""
)

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=791)
    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure", min_value=0.0, max_value=10.0, value=3.30, step=0.01, format="%.2f"
    )
    fuel_pressure = st.number_input(
        "Fuel Pressure", min_value=0.0, max_value=25.0, value=6.66, step=0.01, format="%.2f"
    )

with col2:
    coolant_pressure = st.number_input(
        "Coolant Pressure", min_value=0.0, max_value=10.0, value=2.34, step=0.01, format="%.2f"
    )
    lub_oil_temp = st.number_input(
        "Lub Oil Temp (°C)", min_value=60.0, max_value=100.0, value=77.64, step=0.01, format="%.2f"
    )
    coolant_temp = st.number_input(
        "Coolant Temp (°C)", min_value=60.0, max_value=100.0, value=78.43, step=0.01, format="%.2f"
    )

# Engineer temp_differential exactly as prep.py does
temp_differential = coolant_temp - lub_oil_temp

input_data = pd.DataFrame(
    [
        {
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp,
            "temp_differential": temp_differential,
        }
    ]
)[FEATURE_COLS]

if st.button("Predict Engine Condition"):
    maintenance_probability = model.predict_proba(input_data)[:, 1][0]
    prediction = int(model.predict(input_data)[0])
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.warning(f"**Maintenance Required** (probability: {maintenance_probability:.2%})")
    else:
        st.success(f"**Normal Condition** (probability of maintenance: {maintenance_probability:.2%})")
