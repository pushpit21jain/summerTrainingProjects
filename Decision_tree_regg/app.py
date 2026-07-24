import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰")

st.title("💰 Insurance Cost Prediction")
st.write("Enter the customer details below.")

# Load model with error handling
model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' not found. Please run the notebook first to train the model.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)

sex = st.selectbox("Sex", ["male", "female"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Encoding
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3,
}
region = region_map[region]

# Create DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region],

})

# Prediction
if st.button("Predict Insurance Cost"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Insurance Cost: ₹{prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")