import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model/crop_recommender_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.title("🌱 Crop Suitability Recommender for Sri Lanka")
st.write("Enter environmental conditions and click Predict to get the most suitable crop.")

# Sidebar sliders
st.sidebar.header("Input Parameters")
N = st.sidebar.slider("Nitrogen (N)", 0, 200, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 200, 50)
K = st.sidebar.slider("Potassium (K)", 0, 200, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 500, 100)

# Collect inputs
input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
input_scaled = scaler.transform(input_data)

# Predict only when button pressed
if st.button("🔍 Predict Crop"):
    prediction = model.predict(input_scaled)
    crop = label_encoder.inverse_transform(prediction)[0]

    st.subheader("Recommended Crop 🌾")
    st.success(f"The most suitable crop is: **{crop}**")

    st.subheader("Your Input Parameters")
    st.table({
        "N": [N],
        "P": [P],
        "K": [K],
        "Temperature": [temperature],
        "Humidity": [humidity],
        "pH": [ph],
        "Rainfall": [rainfall],
    })
