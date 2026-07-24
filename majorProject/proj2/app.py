import streamlit as st
import pandas as pd
import joblib

# =====================================
# Load Model and Encoders
# =====================================

model = joblib.load("music_listener_model.pkl")
encoders = joblib.load("encoders.pkl")

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Music Addiction Predictor",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Music Addiction Prediction")
st.write("Predict whether a person is a Heavy Listener or a Normal Listener.")

st.divider()

# =====================================
# User Inputs
# =====================================

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=20
)

streaming = st.selectbox(
    "Primary Streaming Service",
    encoders["Primary streaming service"].classes_
)

while_working = st.selectbox(
    "Listen While Working?",
    encoders["While working"].classes_
)

instrumentalist = st.selectbox(
    "Instrumentalist",
    encoders["Instrumentalist"].classes_
)

composer = st.selectbox(
    "Composer",
    encoders["Composer"].classes_
)

genre = st.selectbox(
    "Favorite Genre",
    encoders["Fav genre"].classes_
)

exploratory = st.selectbox(
    "Exploratory",
    encoders["Exploratory"].classes_
)

foreign = st.selectbox(
    "Foreign Language Music",
    encoders["Foreign languages"].classes_
)

bpm = st.number_input(
    "BPM",
    min_value=40,
    max_value=250,
    value=120
)


# =====================================
# Prediction
# =====================================

if st.button("Predict"):

    data = pd.DataFrame({
        "Age": [age],

        "Primary streaming service": [
            encoders["Primary streaming service"].transform([streaming])[0]
        ],

        "While working": [
            encoders["While working"].transform([while_working])[0]
        ],

        "Instrumentalist": [
            encoders["Instrumentalist"].transform([instrumentalist])[0]
        ],

        "Composer": [
            encoders["Composer"].transform([composer])[0]
        ],

        "Fav genre": [
            encoders["Fav genre"].transform([genre])[0]
        ],

        "Exploratory": [
            encoders["Exploratory"].transform([exploratory])[0]
        ],

        "Foreign languages": [
            encoders["Foreign languages"].transform([foreign])[0]
        ],

        "BPM": [bpm],

        "Anxiety": [5],

        "Depression": [5],

        "Insomnia": [5],

        "OCD": [5]
    })

    # =====================================
    # Match Feature Order Used During Training
    # =====================================

    data = data[model.feature_names_in_]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)

    st.divider()

    if prediction == 1:
        st.success("🎧 Heavy Music Listener")
    else:
        st.info("🎵 Normal Music Listener")

    st.write("### Prediction Probability")

    st.write(
        f"Heavy Listener : **{probability[0][1]*100:.2f}%**"
    )

    st.write(
        f"Normal Listener : **{probability[0][0]*100:.2f}%**"
    )

    st.divider()

    st.subheader("Input Summary")

    st.dataframe(data)