import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Salary Prediction",
    layout="centered"
)


model = joblib.load("model.pkl")  


st.title("Employee Salary Prediction")
st.write("Enter the employee's years of experience to predict the salary.")

st.divider()

years = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)


if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "YearsExperience": [years]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")

st.divider()

st.caption("Built using Streamlit & Scikit-learn")