import joblib,os
from PIL import Image
import numpy as np
from tensorflow import keras

class DigitRecognizer:
    def __init__(self):
        self.model = keras.models.load_model("digit_recognizer.keras")
    
        
    def predict(self, image):
        image = image.convert("L")            # Convert to grayscale
        image = image.resize((28, 28))        # Resize to MNIST size (28×28)
        
        image = np.array(image, dtype=np.float32) / 255.0       # Convert to numpy array and normalize to [0,1]
        
        image = image.reshape(1, 784)           # Flatten to (1, 784)
        probs = self.model.predict(image, verbose=0)    # Get probabilities (suppress progress output)
        
        return np.argmax(probs, axis=1)