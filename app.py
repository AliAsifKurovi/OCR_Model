import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st.title("🧠 OCR - Handwritten Letter Recognition")

#model = tf.keras.models.load_model("ocr_model.h5")
model = tf.keras.models.load_model("ocr_model.keras")
model = tf.keras.models.load_model("final_ocr_model.keras")

uploaded_file = st.file_uploader("📤 Upload a 28x28 grayscale image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_char = chr(predicted_class + ord('A'))

    st.success(f"🎯 Predicted Character: {predicted_char}")

    # Show top 3 predictions
    top_3 = prediction[0].argsort()[-3:][::-1]
    st.subheader("🔢 Top 3 Predictions:")
    for i in top_3:
        char = chr(i + ord('A'))
        confidence = prediction[0][i] * 100
        st.write(f"{char}: {confidence:.2f}%")

    # Show processed image
    st.subheader("🔍 Processed Image Sent to Model")
    fig, ax = plt.subplots()
    ax.imshow(img_array.squeeze(), cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
