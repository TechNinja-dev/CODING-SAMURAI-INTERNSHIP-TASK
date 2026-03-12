# ✍️ Handwritten Digit Recognizer

A web application that predicts handwritten digits (0–9) using a neural network trained on the MNIST dataset. Users can draw a digit on an interactive canvas and get an instant prediction.

This project demonstrates an **end-to-end machine learning workflow**, including:

- Data loading and preprocessing
- Neural network design and training with TensorFlow/Keras
- Model evaluation and visualization
- Model saving for deployment
- Interactive web interface using Streamlit

---

# 📌 Project Overview

This project builds a **neural network classification model** to recognize **handwritten digits** from the MNIST dataset.

The workflow follows standard **deep learning development practices** using **TensorFlow/Keras, data normalization, and model evaluation techniques**.

The project includes:

- Loading and exploring the MNIST dataset
- Normalizing pixel values and flattening images
- Building a feedforward neural network
- Training the model with validation split
- Evaluating accuracy on test data
- Visualizing training history (loss and accuracy curves)
- Saving the trained model for deployment
- Building an interactive Streamlit app for real‑time digit recognition

---

# 📊 Dataset

Dataset used in this project:  
👉 **MNIST** – Modified National Institute of Standards and Technology database  
The dataset is included in `tensorflow.keras.datasets`.

The dataset contains 70,000 grayscale images of handwritten digits (0–9), split into 60,000 training and 10,000 test images.

Typical properties of the dataset:

| Property | Description |
|--------|-------------|
| Image size | 28 × 28 pixels |
| Color | Grayscale (0–255) |
| Labels | Digit 0 through 9 |

The dataset is widely used as a **benchmark for image classification tasks**.

---

# 🎯 Target Classes

The model predicts the following digit classes:

| Digit |
|------|
| 0 |
| 1 |
| 2 |
| 3 |
| 4 |
| 5 |
| 6 |
| 7 |
| 8 |
| 9 |

---

# 📝 Dataset Features

| Feature | Description |
|-------|-------------|
| Pixel values | 784 features (28×28 flattened) |

Each image is transformed into a **784‑dimensional numerical vector**, with pixel intensities normalized to the range **[0, 1]** for better neural network training.

---

# ⚙️ Machine Learning Workflow

The complete ML pipeline follows this workflow:

```
MNIST Dataset
    ↓
Normalize (0–1) & Flatten (784)
    ↓
Train / Validation Split
    ↓
Build Neural Network
    ↓
Train Model
    ↓
Evaluate on Test Set
    ↓
Save Model
    ↓
Deploy in Streamlit App
```

---

# 🧠 Model Architecture

The neural network consists of:
```
Input Layer (784 neurons)
    ↓
Dense Layer (128 neurons, ReLU activation)
    ↓
Dense Layer (64 neurons, ReLU activation)
    ↓
Output Layer (10 neurons, Softmax activation)
```

- **ReLU** activation introduces non‑linearity.
- **Softmax** outputs a probability distribution over the 10 digit classes.
- **Adam** optimizer is used for efficient training.
- **Loss function**: Sparse Categorical Crossentropy (suitable for integer labels).

This architecture balances **simplicity and accuracy** for the MNIST task.

---

# 🔧 Feature Engineering

Minimal feature engineering is required for MNIST:

| Step | Description |
|------|-------------|
| Normalization | Scale pixel values from 0–255 to 0–1 (helps gradient descent) |
| Flattening | Convert 28×28 images to 1D arrays of length 784 |

No hand‑crafted features are needed – the neural network learns features directly from pixels.

---

# 📈 Model Performance

The model achieves high accuracy on the test set:

| Metric | Value |
|--------|-------|
| Test Accuracy | ~98% (depending on training) |

Additional evaluation:

- **Confusion Matrix** can be generated to see per‑class performance.
- **Training/Validation loss and accuracy curves** are plotted to monitor overfitting.

---

# 📊 Visualizations Included

The training notebook includes:

- Sample images from the dataset with their labels
- Loss curves (training vs. validation)
- Accuracy curves (training vs. validation)

These visualizations help understand **model convergence and potential overfitting**.

---

# 💾 Model Saving

After training, the model is saved using Keras’ built‑in `save()` method.

Saved file:  
```
digit_recognizer.keras
```

This file contains the **architecture, weights, and optimizer state** – ready for deployment without retraining.

---

# 🗂️ Project Structure
```
Handwritten-Digit-Recognizer
│
├── app.py # Streamlit web application
├── app4.ipynb
├── deployment_config.py # DigitRecognizer class (loads model, preprocesses)
├── digit_recognizer.keras # Trained Keras model
├── README.md # This file
```

---

# 🚀 Running the Project

## 1️⃣ Install Required Libraries
```
pip install streamlit streamlit-drawable-canvas tensorflow pillow numpy
```

---

# 🧠 Train the Model

Open the notebook and run all cells to:

- load the MNIST dataset
- preprocess the data
- build and train the neural network
- save the model as `digit_recognizer.keras`

---

# 🌐 Run the Streamlit App
```
streamlit run app.py
```

# 🧪 Example Prediction Workflow
```
User draws a digit on canvas
    ↓
Convert canvas to grayscale, resize to 28×28
    ↓
Normalize pixel values (0–1) and flatten to 784
    ↓
Neural Network Model
    ↓
Predicted Digit
```

**Example:**

| Drawing | Preprocessed (28×28) | Prediction |
|---------|------------------------|------------|
| (a hand‑drawn '5') | (28×28 grayscale image) | **5** |

---

# 🎯 Key Features of the Project

✔ End-to-end deep learning workflow with TensorFlow/Keras  
✔ Real‑time digit recognition via Streamlit + drawable canvas  
✔ Preprocessing that matches training (normalization, flattening)  
✔ Clean separation of model loading and prediction logic  
✔ Side‑by‑side display of preprocessed image and predicted digit  
✔ Model saving for easy deployment  

---

# 🧑‍💻 Author

**Prakhar Srivastava**

Python Developer | Machine Learning Enthusiast
