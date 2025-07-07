# 🧠 Optical Character Recognition (OCR) with TensorFlow and EMNIST

This project implements a complete OCR system that recognizes handwritten English alphabet letters (A–Z) using a Convolutional Neural Network (CNN) built from scratch with TensorFlow. The EMNIST Letters dataset is used for training, and custom character images can be used for prediction using the Pillow library. 
This has web interface using Streamlit
---

## 📂 Project Structure

OCR_Model/ <br>
├── train_model.py # Trains CNN on EMNIST and saves model <br>
├── app.py # redirect to web page Loads image and predicts character <br>
├── ocr_model.h5 # Trained Keras model (generated after training) <br>
├── ocr_model.keras # Trained Keras model (generated after training) <br>
├── best_ocr_model.keras # best Trained Keras model (generated after training) <br>
├── final_ocr_model.keras # choose this as final for app best Trained Keras model (generated after training) <br>
└── README.md # Project documentation <br>

## 🔧 Requirements
- Python 3.8–3.11 recommended
- pip (Python package installer)

### 🧪 Install Dependencies
```bash
pip install tensorflow tensorflow-datasets pillow numpy
```

### 🚀 How to Train the Model
Run this script to train a CNN model on the EMNIST letters dataset:
```bash
python train_model.py      # trains and saves model
```
This will:
- Load the EMNIST dataset
- Normalize and preprocess the data
- Train a CNN model
- Save the model to ocr_model.h5

### 🧪 How to Predict a Character
Once trained, you can predict a character from an image using:
```bash
streamlit run app.py          # launches web app
```

#### 🖼️ Image Requirements:
- Grayscale image (.jpg or .png)
- Single handwritten letter centered in the image
- File name: test_char.jpg (or modify in predict.py)
- Image will be resized automatically to 28x28 pixels

### 🧠 Model Architecture
- Input: 28x28 grayscale images
- Conv2D → MaxPool → Conv2D → MaxPool → Flatten
- Dense (128 units, ReLU)
- Output: Dense (26 units, Softmax)

 ### 🧠 CNN Model Architecture Overview
- **Input Layer:** 28×28 grayscale image
- **Conv2D Layer 1:** 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling Layer 1:** 2×2 pool size
- **Conv2D Layer 2:** 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling Layer 2:** 2×2 pool size
- **Flatten Layer:** Converts 2D feature maps to 1D vector
- **Dense Layer:** 128 units, ReLU activation
- **Output Layer:** 26 units (for letters A–Z), Softmax activation

### 📊 Dataset
- Name: EMNIST Letters
- Source: TensorFlow Datasets
- Classes: 26 (A–Z)
- Format: 28x28 grayscale handwritten characters

### ✅ Example Output
`Predicted Character: G`

### 📌 To Do / Ideas
 • Extend to multi-character word recognition
 • Improve accuracy with data augmentation

### 📃 License
This project is open-source under the MIT License.

### 🤝 Contributing
Pull requests and contributions are welcome! If you’d like to suggest improvements or report bugs, please open an issue.

### 🙋‍♂️ Author
`Mohammad Ali Asif` <br>
<a href="https://www.linkedin.com/in/mohammad-ali-asif ">linkedIn</a>
 




