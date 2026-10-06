import streamlit as st
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Load trained model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error(
        "Model files not found. Please train the model first "
        "and make sure model.pkl and vectorizer.pkl are in this folder."
    )
    st.stop()

# Title
st.title("📰 AI-Based Fake News & Misinformation Detector")

st.write(
    "Enter a news headline or article below. "
    "The machine-learning model will classify the text."
)

# Input
news_text = st.text_area(
    "📝 Enter News Text",
    height=220,
    placeholder="Paste a news headline or article here..."
)

# Analyze button
if st.button("🔍 Analyze News", use_container_width=True):

    if not news_text.strip():
        st.warning("⚠️ Please enter some news text.")

    else:
        # Convert text into TF-IDF features
        text_features = vectorizer.transform([news_text])

        # Make prediction
        prediction = model.predict(text_features)[0]

        # Get probability
        probabilities = model.predict_proba(text_features)[0]
        confidence = max(probabilities) * 100

        # Display result
        st.subheader("📊 Result")

        prediction_text = str(prediction).lower()

        if prediction_text == "fake":
            st.error("⚠️ LIKELY FAKE / MISLEADING")
        elif prediction_text == "real":
            st.success("✅ LIKELY REAL")
        else:
            st.info(f"Prediction: {prediction}")

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        # Information
        st.info(
            "⚠️ This is an AI/ML prediction based on patterns learned "
            "from the training dataset. It is not definitive proof "
            "that a news article is true or false."
        )

# About section
st.markdown("---")

st.subheader("ℹ️ About the Project")

st.write(
    """
    This project uses Natural Language Processing (NLP) and Machine
    Learning to analyze news text. TF-IDF converts the text into
    numerical features, and a trained classification model predicts
    whether the content is likely real or fake.
    """
)

st.subheader("🛠️ Technologies Used")

st.write(
    "Python • Streamlit • NLP • TF-IDF • Scikit-learn • "
    "Logistic Regression"
)