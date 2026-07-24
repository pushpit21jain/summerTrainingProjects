import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Placement Prediction System")
st.write("Enter the student's CGPA to predict placement.")

# Input
cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=6.0,
    step=0.1
)

# Prediction
if st.button("Predict Placement"):
    data = pd.DataFrame(
        [[cgpa]],
        columns=["cgpa"]
    )

    prediction = model.predict(data)[0]

    # Display result
    if prediction == 1:
        st.success("✅ The student is likely to be Placed.")
    else:
        st.error("❌ The student is not likely to be Placed.")