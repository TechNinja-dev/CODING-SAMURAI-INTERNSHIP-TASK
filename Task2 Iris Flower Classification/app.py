import streamlit as st
import pandas as pd
import joblib

# --------------------------------------
# Page configuration
# --------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# --------------------------------------
# Load trained model
# --------------------------------------
@st.cache_resource
def load_model():
    model, encoder = joblib.load("iris_flower_classifier_model.pkl")
    return model,encoder

model,encoder = load_model()

# --------------------------------------
# Sidebar
# --------------------------------------
with st.sidebar:

    st.title("🌸 Iris Classifier")

    st.markdown("""
### Model Information

**Algorithm:** Random Forest  
**Task:** Flower Classification  

### Features Used

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

### Classes

- Setosa  
- Versicolor  
- Virginica  
""")

    st.divider()

    st.write("Developed by")
    st.write("**Prakhar Srivastava**")

# --------------------------------------
# Main Title
# --------------------------------------
st.title("🌸 Iris Flower Classification System")

st.markdown(
"""
This application predicts the **species of an iris flower** using a trained machine learning model.

Provide the flower measurements below to classify the species.
"""
)

st.divider()

# --------------------------------------
# Input Form
# --------------------------------------
with st.form("iris_form"):

    st.subheader("Flower Measurements")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.1
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.5
        )

    with col2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.4
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=0.2
        )

    submit = st.form_submit_button("Predict Species")

# --------------------------------------
# Prediction
# --------------------------------------
if submit:

    try:

        input_data = pd.DataFrame({
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width]
        })

        prediction = model.predict(input_data)

        predicted_species = encoder.inverse_transform(prediction)[0]

        st.subheader("Prediction Result")

        st.success(f"Predicted Species: **{predicted_species}**")

        st.subheader("Input Summary")

        st.dataframe(input_data)

    except Exception as e:

        st.error("Prediction failed")
        st.exception(e)