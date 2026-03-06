# 💬 Tweet Sentiment Analysis using Logistic Regression

A Natural Language Processing project that classifies tweets into sentiment categories based on the textual content of the tweet.

This project demonstrates an **end-to-end NLP machine learning workflow**, including:

* Text preprocessing
* Feature extraction using TF-IDF
* Model training
* Model evaluation
* Pipeline integration
* Model saving for deployment

---

# 📌 Project Overview

This project builds a **machine learning classification model** to predict the **sentiment of tweets** using a **Logistic Regression classifier**.

The workflow follows standard **Natural Language Processing (NLP) development practices** using **scikit-learn pipelines, TF-IDF vectorization, and model evaluation techniques**.

The project includes:

* Text cleaning and preprocessing
* Feature extraction using TF-IDF
* Model training using Logistic Regression
* Model evaluation using accuracy and classification metrics
* Confusion matrix visualization
* Saving the trained model for future use

---

# 📊 Dataset

Dataset used in this project:
👉 https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

The dataset contains tweets labeled with sentiment categories.

Typical properties of the dataset:

| Property | Description |
|--------|-------------|
| Text | Tweet content |
| Sentiment | Label representing sentiment |

The dataset is commonly used for **sentiment classification tasks in NLP**.

---

# 💬 Target Classes

The model predicts the following sentiment categories:

| Sentiment |
|-----------|
| Positive |
| Negative |

---

# 📝 Dataset Features

| Feature | Description |
|-------|-------------|
| text | The tweet text used for sentiment prediction |

The text is transformed into **numerical features using TF-IDF vectorization**.

---

# ⚙️ Machine Learning Workflow

The complete ML pipeline follows this workflow:

```
Tweet Dataset
   ↓
Text Cleaning & Preprocessing
   ↓
Train Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
```

---

# 🧠 Model Architecture

The model pipeline consists of:

```
Pipeline
   │
   ├── TFIDF Vectorizer
   │
   └── Logistic Regression Classifier
```

Logistic Regression was selected because:

* Performs well on text classification tasks
* Efficient for high-dimensional sparse data
* Works well with TF-IDF features
* Fast training and prediction

---

# 🔧 Feature Engineering

Text features are extracted using **TF-IDF Vectorization**.

Configuration used:

| Parameter | Description |
|----------|-------------|
| max_features | Limits vocabulary size |
| ngram_range | Uses both single words and word pairs |
| stop_words | Removes common English stop words |

TF-IDF converts raw text into **numerical vectors representing word importance**.

---

# 📈 Model Performance

Model evaluation includes:

| Metric | Description |
|------|-------------|
| Accuracy | Overall prediction accuracy |
| Precision | Correct positive predictions |
| Recall | Model's ability to detect positives |
| F1 Score | Balance between precision and recall |

Additional evaluation tools:

* Confusion Matrix
* Classification Report

---

# 📊 Visualizations Included

The notebook includes visualizations to understand model performance:

* Confusion Matrix
* Sentiment distribution in dataset
* Model evaluation metrics

These visualizations help interpret **how well the model distinguishes between sentiment classes**.

---

# 💾 Model Saving

After training, the model is saved using **joblib**.

Saved file:

```
sentiment.pkl
```

This allows the trained model to be **loaded later without retraining**.

---

# 🗂️ Project Structure

```
Sentiment-Analysis-Tweets
│
├── sentiment_analysis.ipynb
├── sentiment.pkl
└── README.md
```

---

# 🚀 Running the Project

## 1️⃣ Install Required Libraries

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

---

# 🧠 Train the Model

Run the notebook to:

* preprocess tweet text
* train the TF-IDF + Logistic Regression model
* evaluate performance
* save the trained model

---

# 🧪 Example Prediction Workflow

```
Tweet Text
   ↓
TF-IDF Vectorizer
   ↓
Logistic Regression Model
   ↓
Predicted Sentiment
```

Example:

```
Input
"I really love the service and the experience!"

Prediction
Positive
```

---

# 🎯 Key Features of the Project

✔ End-to-end NLP machine learning workflow  
✔ TF-IDF feature extraction  
✔ Logistic Regression classifier  
✔ Text preprocessing pipeline  
✔ Model evaluation using classification metrics  
✔ Model saving for future deployment  

---

# 🧑‍💻 Author

**Prakhar Srivastava**

Python Developer | Machine Learning Enthusiast