import streamlit as st
import pandas as pd
import joblib
model = joblib.load("anxiety_prediction_model.pkl")
st.set_page_config(
    page_title="Music Anxiety Prediction",
    page_icon="🎵",
    layout="wide"
)
st.title("🎵 Music Anxiety Prediction")

st.write(
    """
    Predict whether a user has **High Anxiety**
    based on their music listening habits.
    """
)

st.sidebar.title("About")

st.sidebar.info(
"""
Machine Learning Project

Algorithm:
Random Forest

Preprocessing:
ColumnTransformer + OneHotEncoder

Developed using:
Streamlit
"""
)

col1,col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20
    )

    hours = st.slider(
        "Hours per day",
        0.0,
        24.0,
        2.0
    )

    bpm = st.number_input(
        "BPM",
        40,
        250,
        120
    )

with col2:

    streaming = st.selectbox(
        "Streaming Service",
        [
            "Spotify",
            "YouTube Music",
            "Apple Music",
            "Pandora",
            "Other"
        ]
    )

    working = st.selectbox(
        "While Working",
        ["Yes","No"]
    )

    instrumentalist = st.selectbox(
        "Instrumentalist",
        ["Yes","No"]
    )

    composer = st.selectbox(
        "Composer",
        ["Yes","No"]
    )

    exploratory = st.selectbox(
        "Exploratory",
        ["Yes","No"]
    )

    foreign = st.selectbox(
        "Foreign Languages",
        ["Yes","No"]
    )

    effects = st.selectbox(
        "Music Effects",
        [
            "Improve",
            "No effect",
            "Worsen"
        ]
    )

    language = st.selectbox(
        "Music Language",
        [
            "Hindi",
            "English",
            "Punjabi",
            "Tamil",
            "Telugu",
            "Marathi",
            "Bengali"
        ]
    )
if st.button("Predict Anxiety"):

    sample = pd.DataFrame({

        "Age":[age],
        "Primary streaming service":[streaming],
        "Hours per day":[hours],
        "While working":[working],
        "Instrumentalist":[instrumentalist],
        "Composer":[composer],
        "Exploratory":[exploratory],
        "Foreign languages":[foreign],
        "BPM":[bpm],
        "Music effects":[effects],
        "MusicLanguage":[language]
        #'Frequency [Folk]':[2], 'Frequency [Country]':[2], 'Frequency [Rap]':[2], 'Frequency [R&B]':[2], 'Frequency [Gospel]':[2], 'Frequency [Jazz]':[2], 'Frequency [Rock]':[2], 'Frequency [EDM]':[2], 'Frequency [Classical]':[2], 'Frequency [Video game music]':[2], 'Frequency [Lofi]':[2], 'Frequency [Metal]':[2], 'Frequency [K pop]':[2], 'Frequency [Pop]':[2], 'Frequency [Latin]':[2],'Frequency [Hip hop]':[2]
        ,'Insomnia':[5], 'Depression':[5],'OCD':[2]
    })

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0]

    if prediction == 1:
        st.error("⚠️ High Anxiety Detected")
    else:
        st.success("😊 Low Anxiety")

    confidence = max(probability)

    st.metric(
        "Prediction Confidence",
        f"{confidence*100:.2f}%"
    )

    result = pd.DataFrame({
        "Class":["Low Anxiety","High Anxiety"],
        "Probability":probability
    })

 