import streamlit as st
import joblib
import pandas as pd
model = joblib.load("best_model.pkl")
st.set_page_config(page_title="Titanic Prediction", page_icon="🚢")

st.title("🚢 Titanic Prediction App")
st.write("Enter passenger details below.")

pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)

sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)

parch = st.number_input("Parents/Children Aboard", 0, 10, 0)

fare = st.number_input("Fare", min_value=0.0, value=30.0)

travel_class = st.selectbox("Class", [1, 2, 3])

adult_male = st.selectbox("Adult Male", ["Yes", "No"])
adult_male = 1 if adult_male == "Yes" else 0

alone = st.selectbox("Travelling Alone", ["Yes", "No"])
alone = 1 if alone == "Yes" else 0

embark = st.selectbox(
    "Embark Town",
    ["Cherbourg", "Queenstown", "Southampton"]
)

embark_q = 1 if embark == "Queenstown" else 0
embark_s = 1 if embark == "Southampton" else 0

who = st.selectbox(
    "Who",
    ["Man", "Woman", "Child"]
)

who_man = 1 if who == "Man" else 0
who_woman = 1 if who == "Woman" else 0

if st.button("Predict"):

    input_data = pd.DataFrame({
        "pclass": [pclass],
        "sex": [sex],
        "age": [age],
        "sibsp": [sibsp],
        "parch": [parch],
        "fare": [fare],
        "class": [travel_class],
        "adult_male": [adult_male],
        "alone": [alone],
        "embark_town_Queenstown": [embark_q],
        "embark_town_Southampton": [embark_s],
        "who_man": [who_man],
        "who_woman": [who_woman]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Prediction: {prediction:.4f}")

    if prediction >= 0.5:
        st.success("Survived")
    else:
        st.error("Did Not Survive")