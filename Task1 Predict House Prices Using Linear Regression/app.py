# import streamlit as st
# import pandas as pd
# import joblib

# # -------------------------------

# # Page configuration

# # -------------------------------

# st.set_page_config(
# page_title="House Price Prediction",
# page_icon="🏠",
# layout="wide"
# )

# # -------------------------------

# # Load trained model

# # -------------------------------

# @st.cache_resource
# def load_model():
#     model = joblib.load("house_price_model.pkl")
#     return model

# model = load_model()

# # -------------------------------

# # Title

# # -------------------------------

# st.title("🏠 House Price Prediction System")
# st.write(
# "Enter the property details below to estimate the house price using the trained machine learning model."
# )

# st.divider()

# # -------------------------------

# # Input Form

# # -------------------------------

# with st.form("prediction_form"):
#     st.subheader("Property Details")

#     col1, col2 = st.columns(2)

#     with col1:
#         bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
#         bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)
#         living_area = st.number_input("Living Area (m²)", min_value=10.0, value=120.0)
#         grade = st.number_input("House Grade", min_value=1, max_value=10, value=5)
#         year = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)

#     with col2:
#         nice_view = st.selectbox("Nice View", [True, False])
#         perfect_condition = st.selectbox("Perfect Condition", [True, False])
#         renovated = st.selectbox("Renovated", [True, False])
#         single_floor = st.selectbox("Single Floor", [True, False])
#         quartile_zone = st.selectbox("Quartile Zone", [1, 2, 3, 4])

#     # Submit button MUST be inside the form
#     submitted = st.form_submit_button("Predict Price")
    
    


# # -------------------------------

# # Prediction

# # -------------------------------



# if submitted:
#     try:
#         input_data = pd.DataFrame({
#             "bedrooms": [bedrooms],
#             "real_bathrooms": [bathrooms],
#             "living_in_m2": [living_area],
#             "grade": [grade],
#             "year": [year],
#             "nice_view": [nice_view],
#             "perfect_condition": [perfect_condition],
#             "renovated": [renovated],
#             "single_floor": [single_floor],
#             "quartile_zone": [quartile_zone]
#         })

#         prediction = model.predict(input_data)[0]

#         st.subheader("Prediction Result")

#         st.markdown(
#         f"""
#         ### 💰 Estimated Price

#         # **${prediction:,.2f}**

#         This prediction is generated using the trained machine learning model.
#         """
# )

#     except Exception as e:
#         st.error("Prediction failed. Please check the input values.")
#         st.exception(e)



import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
with st.sidebar:
    st.title("🏠 Model Information")

    st.markdown("""
**Model Type:** Linear Regression  
**Preprocessing:**  
- Standard Scaling  
- Column Transformer  
- Pipeline Integration  

**Features Used**
- Bedrooms
- Bathrooms
- Living Area
- Grade
- Year Built
- Nice View
- Renovated
- Perfect Condition
- Single Floor
- Quartile Zone
""")

    st.divider()

    st.subheader("Developer")
    st.write("Prakhar Srivastava")

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.title("🏡 House Price Prediction Dashboard")

st.markdown(
"""
This application predicts **house prices** using a trained **Machine Learning model**.  
Fill in the property details below and click **Predict Price** to estimate the value.
"""
)

st.divider()

# -------------------------------------------------------
# Input Form
# -------------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Property Features")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)
        living_area = st.number_input("Living Area (m²)", min_value=10.0, value=120.0)
        grade = st.number_input("House Grade", min_value=1, max_value=10, value=5)
        year = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)

    with col2:
        nice_view = st.selectbox("Nice View", [True, False])
        perfect_condition = st.selectbox("Perfect Condition", [True, False])
        renovated = st.selectbox("Renovated", [True, False])
        single_floor = st.selectbox("Single Floor", [True, False])
        quartile_zone = st.selectbox("Quartile Zone", [1, 2, 3, 4])

    submitted = st.form_submit_button("Predict Price")

# -------------------------------------------------------
# Prediction Section
# -------------------------------------------------------
if submitted:

    try:
        input_data = pd.DataFrame({
            "bedrooms": [bedrooms],
            "real_bathrooms": [bathrooms],
            "living_in_m2": [living_area],
            "grade": [grade],
            "year": [year],
            "nice_view": [nice_view],
            "perfect_condition": [perfect_condition],
            "renovated": [renovated],
            "single_floor": [single_floor],
            "quartile_zone": [quartile_zone]
        })

        prediction = model.predict(input_data)[0]

        st.subheader("💰 Prediction Result")
        st.success(f"Estimated House Price: {prediction:,.2f}")

        st.subheader("Input Summary")
        st.dataframe(input_data)

    except Exception as e:
        st.error("Prediction failed. Please check the input values.")
        st.exception(e)