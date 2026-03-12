import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from deployment_config import DigitRecognizer
import tempfile

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return DigitRecognizer()

model = load_model()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

# ----------------------------
# Sidebar (Like Reference UI)
# ----------------------------
st.sidebar.title("✍️ Digit Recognizer")

st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Neural Network")
st.sidebar.write("Dataset: MNIST")
st.sidebar.write("Task: Digit Classification")

st.sidebar.header("Features Used")
st.sidebar.markdown("""
- Handwritten Image
- 28 x 28 Pixel Grayscale
- Flattened Pixel Values
""")

st.sidebar.header("Classes")
st.sidebar.markdown("""
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by")
st.sidebar.write("Prakhar Srivastava")

# ----------------------------
# Main Title
# ----------------------------
st.title("✍️ Handwritten Digit Recognition System")

st.write(
"""
This application predicts handwritten digits using a trained **MNIST Machine Learning model**.

Draw a digit in the canvas below and the system will classify it.
"""
)

st.markdown("---")

# ----------------------------
# Canvas Section
# ----------------------------
st.subheader("Draw a Digit")

# Create two columns: left for canvas, right for output
col1, col2 = st.columns([1, 1])

with col1:
    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_clicked = st.button("Predict Digit")

with col2:
    # This area will show the processed image and prediction
    if predict_clicked and canvas_result.image_data is not None:
        img_data = canvas_result.image_data

        # Quick blank check using the red channel (white strokes)
        grayscale_check = img_data[:, :, 0]
        if np.mean(grayscale_check) < 5:
            st.warning("Please draw a digit first!")
        else:
            # Convert to PIL and preprocess for the model
            img_pil = Image.fromarray(img_data.astype("uint8")).convert("L")  # grayscale
            img_resized = img_pil.resize((28, 28))                           # model input size

            # Show the preprocessed image (28×28)
            st.image(img_resized, caption="Preprocessed Image (28×28)", width=150)

            # Predict using the model
            # (Assuming your DigitRecognizer.predict does NOT invert colors)
            prediction = model.predict(img_resized)   # or model.predict(img_pil) if it handles resizing
            st.success(f"Predicted Digit: {prediction[0]}")