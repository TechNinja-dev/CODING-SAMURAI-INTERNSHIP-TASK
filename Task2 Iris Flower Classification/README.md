# 🌸 Iris Flower Classification using Random Forest

A Machine Learning project that classifies iris flowers into different species based on flower measurements such as sepal length, sepal width, petal length, and petal width.

This project demonstrates an **end-to-end machine learning workflow**, including:

* Data preprocessing
* Model training
* Hyperparameter tuning
* Model evaluation
* Pipeline integration
* Interactive **Streamlit UI deployment**

---

# 📌 Project Overview

This project builds a **machine learning classification model** to predict the species of an iris flower using a **Random Forest Classifier**.

The workflow follows standard ML development practices using **scikit-learn pipelines, hyperparameter tuning, and model deployment with Streamlit**.

The project includes:

* Data exploration and visualization
* Model pipeline creation
* Hyperparameter tuning using GridSearchCV
* Model evaluation using accuracy and confusion matrix
* Interactive **web-based prediction interface**

---

# 📊 Dataset

Dataset used in this project:

👉 https://www.kaggle.com/datasets/himanshunakrani/iris-dataset

Dataset characteristics:

| Property          | Value |
| ----------------- | ----- |
| Total samples     | 150   |
| Features          | 4     |
| Classes           | 3     |
| Samples per class | 50    |

---

# 🌼 Target Classes

The model predicts the following iris species:

| Class      |
| ---------- |
| Setosa     |
| Versicolor |
| Virginica  |

---

# 📐 Dataset Features

| Feature      | Description         |
| ------------ | ------------------- |
| sepal_length | Length of the sepal |
| sepal_width  | Width of the sepal  |
| petal_length | Length of the petal |
| petal_width  | Width of the petal  |

All features are **numerical measurements in centimeters**.

---

# ⚙️ Machine Learning Workflow

The complete ML pipeline follows this workflow:

```
Dataset
   ↓
Data Exploration
   ↓
Train Test Split
   ↓
Pipeline Creation
   ↓
Hyperparameter Tuning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Web Application
```

---

# 🧠 Model Architecture

The model pipeline consists of:

```
Pipeline
   │
   └── RandomForestClassifier
```

Random Forest was selected because:

* It performs well on structured datasets
* Handles nonlinear relationships
* Requires minimal preprocessing
* Provides feature importance insights

---

# 🔧 Hyperparameter Tuning

Hyperparameter tuning is performed using **GridSearchCV**.

Parameters tuned include:

| Parameter         | Description                            |
| ----------------- | -------------------------------------- |
| n_estimators      | Number of trees in the forest          |
| max_depth         | Maximum depth of the tree              |
| min_samples_split | Minimum samples required to split      |
| min_samples_leaf  | Minimum samples required at leaf nodes |

The best parameters are selected using **5-fold cross-validation**.

---

# 📈 Model Performance

Typical performance achieved:

| Metric   | Value        |
| -------- | ------------ |
| Accuracy | ~0.95 – 1.00 |

The Iris dataset is well structured, allowing high accuracy with tree-based models.

---

# 📊 Visualizations Included

The notebook includes multiple visualizations to understand the dataset:

* Pairplot of features
* Feature correlation heatmap
* Feature distribution histograms
* Petal length vs petal width scatter plot
* Feature importance plot
* Confusion matrix

These help analyze feature relationships and model performance.

---

# 🌐 Streamlit Web Application

A **Streamlit web interface** is implemented to allow users to interactively predict the iris flower species.

The UI allows users to input:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The trained model then predicts the corresponding flower species.

### Features of the Web App

✔ Clean and interactive UI
✔ Real-time predictions
✔ Input validation
✔ Model loading with caching
✔ Displays prediction result instantly

---

# 🗂️ Project Structure

```
Iris-Flower-Classifier
│
├── app.py
├── iris_classifier.py
├── iris_model.pkl
├── iris_analysis.ipynb
└── README.md
```

---

# 🚀 Running the Project

## 1️⃣ Install Required Libraries

```
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

---

# 🧠 Train the Model

Run the classifier script to train and save the model.

```
python iris_classifier.py
```

This script will:

* preprocess the dataset
* train the Random Forest model
* perform hyperparameter tuning
* save the trained model

---

# 🌸 Run the Streamlit Application

Launch the interactive UI using:

```
streamlit run app.py
```

The application will open in your browser.

---

# 🧪 Example Prediction Workflow

```
Flower Measurements
   ↓
Random Forest Model
   ↓
Predicted Iris Species
```

Example:

```
Input
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

Prediction
Setosa
```

---

# 🎯 Key Features of the Project

✔ End-to-end ML classification workflow
✔ Hyperparameter tuning using GridSearchCV
✔ Random Forest classifier implementation
✔ Pipeline-based model architecture
✔ Interactive Streamlit UI
✔ Feature importance analysis

---

# 🧑‍💻 Author

**Prakhar Srivastava**

Python Developer | Machine Learning Enthusiast

---

