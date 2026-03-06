import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# Load pipeline
model = joblib.load("sentiment.pkl")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("💬 Sentiment Analyzer")

st.sidebar.header("Model Information")

st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Task: Tweet Sentiment Classification")

st.sidebar.header("Features Used")

st.sidebar.markdown("""
- Tweet Text
- TF-IDF Vectorization
- N-grams (Unigram + Bigram)
""")

st.sidebar.header("Classes")

st.sidebar.markdown("""
- Positive
- Negative
""")

st.sidebar.markdown("---")

st.sidebar.write("Developed by")
st.sidebar.write("**Prakhar Srivastava**")

# ---------------- MAIN PAGE ---------------- #

st.title("💬 Tweet Sentiment Analysis System")

st.write(
"""
This application predicts the **sentiment of a tweet**
using a trained Natural Language Processing model.
"""
)

st.write(
"""
Enter a tweet below and the system will classify the sentiment.
"""
)

st.divider()

# ---------------- INPUT SECTION ---------------- #

st.subheader("Tweet Input")

tweet = st.text_area(
    "Enter Tweet",
    height=150,
    placeholder="Example: I absolutely love this phone!"
)

col1, col2, col3 = st.columns([1,1,4])

with col1:
    predict_btn = st.button("Predict Sentiment")

# ---------------- PREDICTION ---------------- #

if predict_btn:

    if tweet.strip() == "":
        st.warning("Please enter a tweet first.")

    else:
        prediction = model.predict([tweet])[0]

        st.subheader("Prediction Result")

        if prediction == "positive":
            st.success("😊 Positive Sentiment")

        elif prediction == "negative":
            st.error("😡 Negative Sentiment")

        else:
            st.info("😐 Neutral Sentiment")