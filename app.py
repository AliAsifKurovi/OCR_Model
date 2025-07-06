# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("🧠 OCR - Handwritten Letter Recognition")

model = tf.keras.models.load_model("ocr_model.h5")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_char = chr(predicted_class + ord('A'))

    st.success(f"Predicted Character: {predicted_char}")
