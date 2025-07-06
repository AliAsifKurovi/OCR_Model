import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('ocr_model.h5')

# Preprocess input image
def preprocess_image(path):
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))           # Resize to 28x28 pixels
    img = np.array(img) / 255.0          # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)       # Reshape for model input
    return img

# Predict character
img = preprocess_image("test_char_2.jpeg")  # Make sure this file exists
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# EMNIST Letters: labels 0 = A, ..., 25 = Z
predicted_char = chr(predicted_class + ord('A'))

print(f"Predicted Character: {predicted_char}")
