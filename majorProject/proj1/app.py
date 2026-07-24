import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "mxmh_survey_results_indian_music.csv"
TARGET_COLUMN = "Fav genre"

FEATURE_COLUMNS = [
    "Age",
    "Primary streaming service",
    "Hours per day",
    "While working",
    "Instrumentalist",
    "Composer",
    "Exploratory",
    "Foreign languages",
    "BPM",
    "Frequency [Classical]",
    "Frequency [Country]",
    "Frequency [EDM]",
    "Frequency [Folk]",
    "Frequency [Gospel]",
    "Frequency [Hip hop]",
    "Frequency [Jazz]",
    "Frequency [K pop]",
    "Frequency [Latin]",
    "Frequency [Lofi]",
    "Frequency [Metal]",
    "Frequency [Pop]",
    "Frequency [R&B]",
    "Frequency [Rap]",
    "Frequency [Rock]",
    "Frequency [Video game music]",
    "Anxiety",
    "Depression",
    "Insomnia",
    "OCD",
    "Music effects",
    "MusicLanguage",
]


@st.cache_resource
def load_trained_model():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["Timestamp", "Permissions", "MusicName", "Singer"], errors="ignore")
    data = data.dropna(subset=[TARGET_COLUMN]).copy()

    for column in data.select_dtypes(include="number").columns:
        data[column] = data[column].fillna(data[column].median())

    for column in data.select_dtypes(include=["object", "string"]).columns:
        mode_values = data[column].mode(dropna=True)
        if not mode_values.empty:
            data[column] = data[column].fillna(mode_values.iloc[0])

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    return model, accuracy, data


model, model_accuracy, training_data = load_trained_model()

st.set_page_config(
    page_title="Music Genre Prediction",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Genre Prediction System")
st.write("Predict a user's favorite music genre based on listening habits.")

st.caption(f"Model validation accuracy: {model_accuracy:.2%}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=20
    )

    hours = st.slider(
        "Hours of Music per Day",
        0.0,
        24.0,
        2.0
    )

    bpm = st.number_input(
        "Average BPM",
        min_value=40,
        max_value=250,
        value=120
    )

    anxiety = st.slider("Anxiety",0,10,5)
    depression = st.slider("Depression",0,10,5)
    insomnia = st.slider("Insomnia",0,10,5)
    ocd = st.slider("OCD",0,10,5)

with col2:

    streaming = st.selectbox(
        "Streaming Service",
        sorted(training_data["Primary streaming service"].dropna().astype(str).unique())
    )

    working = st.selectbox(
        "Listen While Working",
        sorted(training_data["While working"].dropna().astype(str).unique())
    )

    instrumentalist = st.selectbox(
        "Instrumentalist",
        sorted(training_data["Instrumentalist"].dropna().astype(str).unique())
    )

    composer = st.selectbox(
        "Composer",
        sorted(training_data["Composer"].dropna().astype(str).unique())
    )

    exploratory = st.selectbox(
        "Exploratory Listener",
        sorted(training_data["Exploratory"].dropna().astype(str).unique())
    )

    foreign = st.selectbox(
        "Foreign Language Songs",
        sorted(training_data["Foreign languages"].dropna().astype(str).unique())
    )

    effects = st.selectbox(
        "Music Effect",
        sorted(training_data["Music effects"].dropna().astype(str).unique())
    )

    language = st.selectbox(
        "Preferred Music Language",
        sorted(training_data["MusicLanguage"].dropna().astype(str).unique())
    )

st.markdown("---")

st.subheader("Genre Listening Frequency")

genres = [
    "Frequency [Classical]",
    "Frequency [Country]",
    "Frequency [EDM]",
    "Frequency [Folk]",
    "Frequency [Gospel]",
    "Frequency [Hip hop]",
    "Frequency [Jazz]",
    "Frequency [K pop]",
    "Frequency [Latin]",
    "Frequency [Lofi]",
    "Frequency [Metal]",
    "Frequency [Pop]",
    "Frequency [R&B]",
    "Frequency [Rap]",
    "Frequency [Rock]",
    "Frequency [Video game music]"
]

freq = {}

cols = st.columns(4)

for i, g in enumerate(genres):
    with cols[i % 4]:
        freq[g] = st.slider(
            g,
            0,
            5,
            2
        )

st.markdown("---")

if st.button("Predict Genre", use_container_width=True):

    data = {
        "Age":[age],
        "Primary streaming service":[streaming],
        "Hours per day":[hours],
        "While working":[working],
        "Instrumentalist":[instrumentalist],
        "Composer":[composer],
        "Exploratory":[exploratory],
        "Foreign languages":[foreign],
        "BPM":[bpm],
        "Anxiety":[anxiety],
        "Depression":[depression],
        "Insomnia":[insomnia],
        "OCD":[ocd],
        "Music effects":[effects],
        "MusicLanguage":[language]
    }

    for g in genres:
        data[g] = [freq[g]]

    input_df = pd.DataFrame(data)

    prediction = model.predict(input_df)[0]

    st.success(f"🎧 Predicted Favorite Genre: **{prediction}**")