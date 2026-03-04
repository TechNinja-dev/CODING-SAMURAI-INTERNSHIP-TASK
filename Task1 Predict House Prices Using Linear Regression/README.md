# 🏠 House Price Prediction using Linear Regression

A Machine Learning project that predicts house prices based on various property features such as bedrooms, bathrooms, living area, and location indicators.
The project demonstrates an **end-to-end ML workflow**, including **data preprocessing, model training, pipeline integration, evaluation, model persistence, and UI deployment using Streamlit**.

---

# 📌 Project Overview

This project builds a **machine learning model** to estimate house prices using **Linear Regression** with a proper **scikit-learn Pipeline**.

The project also includes:

* Data preprocessing and feature engineering
* Outlier detection and removal
* Model training with **target transformation**
* Model evaluation using **RMSE and R²**
* Model saving using **Joblib**
* Interactive **Streamlit UI** for predictions

---

# 📊 Dataset

Dataset used in this project:

👉 https://www.kaggle.com/datasets/aravinii/house-price-prediction-treated-dataset

The dataset contains features describing house properties such as:

| Feature           | Description                         |
| ----------------- | ----------------------------------- |
| bedrooms          | Number of bedrooms                  |
| bathrooms         | Number of bathrooms                 |
| living_in_m2      | Living area in square meters        |
| grade             | Quality of house construction       |
| nice_view         | Whether the house has a good view   |
| renovated         | Whether the house was renovated     |
| perfect_condition | Property condition indicator        |
| single_floor      | Indicates if house has single floor |
| quartile_zone     | Location price category             |
| date              | Date of sale                        |
| price             | Target variable (house price)       |

---

# ⚙️ Machine Learning Workflow

The ML workflow followed in this project:

```
Dataset
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Outlier Detection (IQR)
   ↓
Train Test Split
   ↓
Pipeline Creation
   ↓
Target Transformation (Log)
   ↓
Model Training
   ↓
Evaluation (RMSE + R²)
   ↓
Model Saving (Joblib)
   ↓
Streamlit UI Deployment
```

---

# 🧠 Model Architecture

The model pipeline consists of:

```
ColumnTransformer
    │
    ├── StandardScaler (numerical features)
    └── OrdinalEncoder (boolean features)
            │
            ↓
      Linear Regression
            │
            ↓
TransformedTargetRegressor
(Log transform target variable)
```

---

# 📈 Model Performance

| Metric   | Value    |
| -------- | -------- |
| RMSE     | ~105,869 |
| R² Score | ~0.724   |

The model explains approximately **72% of variance in house prices**.

---

# 🗂️ Project Structure

```
House-Price-Prediction
│
├── app.py
├── deployment_config.py
├── house_price_model.pkl
├── df_test.csv
├── app1.ipynb
└── README.md
```

---

# 🧹 Data Preprocessing

The following preprocessing steps are applied:

* Convert `date` to `year`
* Drop unnecessary columns
* Detect and remove outliers using **IQR**
* Separate **features (X)** and **target (y)**
* Apply **log transformation** on the target variable

---

# 🧪 Model Training

The training pipeline uses:

* `StandardScaler`
* `OrdinalEncoder`
* `LinearRegression`
* `TransformedTargetRegressor`

This ensures **consistent preprocessing during inference**.

---

# 💾 Model Saving

The trained model is saved using **Joblib**.

```python
import joblib

joblib.dump(model, "house_price_model.pkl")
```

---

# 🚀 Running the Project

## 1️⃣ Install Required Libraries

```
pip install pandas numpy scikit-learn streamlit joblib
```

---

# 🧠 Train the Model

Run the deployment script to train and save the model.

```
python deployment_config.py
```

This will:

* preprocess the data
* train the model
* save the trained model

---

# 🖥️ Run the Streamlit Application

Launch the UI with:

```
streamlit run app.py
```

This will start the web interface for predicting house prices.

---

# 🧾 Example Prediction Workflow

User enters property details:

```
Bedrooms
Bathrooms
Living Area
House Grade
Year Built
View Quality
Condition
Renovation Status
Single Floor Indicator
Quartile Zone
```

The app then:

```
User Input
   ↓
Pipeline Preprocessing
   ↓
Model Prediction
   ↓
House Price Output
```

---

# 🎯 Key Features

✔ End-to-end ML pipeline
✔ Outlier detection using IQR
✔ Log transformation of target variable
✔ ColumnTransformer preprocessing
✔ Model persistence using Joblib
✔ Interactive Streamlit UI
✔ Modular class-based ML architecture

---

# 🧑‍💻 Author

**Prakhar Srivastava**
prakharsrivastava019@gmail.com
Artificial Intelligence and Machine Learning Enthusiast | Python Developer

---

