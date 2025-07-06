# 🧠 Optical Character Recognition (OCR) with TensorFlow and EMNIST

This project implements a complete OCR system that recognizes handwritten English alphabet letters (A–Z) using a Convolutional Neural Network (CNN) built from scratch with TensorFlow. The EMNIST Letters dataset is used for training, and custom character images can be used for prediction using the Pillow library.

---

## 📂 Project Structure

OCR_Model/ <br>
├── train_model.py # Trains CNN on EMNIST and saves model <br>
├── predict.py # Loads image and predicts character <br>
├── ocr_model.h5 # Trained Keras model (generated after training) <br>
├── test_char.jpg # Input image for prediction <br>
├── venv/ # Virtual environment (optional) <br>
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
python train_model.py
```
This will:
- Load the EMNIST dataset
- Normalize and preprocess the data
- Train a CNN model
- Save the model to ocr_model.h5

### 🧪 How to Predict a Character
Once trained, you can predict a character from an image using:
```bash
python predict.py
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

### 📊 Dataset
- Name: EMNIST Letters
- Source: TensorFlow Datasets
- Classes: 26 (A–Z)
- Format: 28x28 grayscale handwritten characters

### ✅ Example Output
`Predicted Character: G`
