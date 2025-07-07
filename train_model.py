import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Load and preprocess dataset
# -------------------------------
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Normalize and adjust labels (1-26 → 0-25)
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label - 1

ds_train = ds_train.map(normalize_img)
ds_test = ds_test.map(normalize_img)

# -------------------------------
# 2. Data Augmentation
# -------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

def augment(image, label):
    return data_augmentation(image), label

ds_train = ds_train.map(augment).cache().shuffle(10000).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(128).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# 3. Build Model
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# -------------------------------
# 4. Callbacks
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_ocr_model.keras", save_best_only=True, monitor='val_accuracy')

# -------------------------------
# 5. Train Model
# -------------------------------
model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=30,
    callbacks=[early_stop, checkpoint]
)

# Save final version
model.save("final_ocr_model.keras")
print("✅ Final model saved as final_ocr_model.keras")

# -------------------------------
# 6. Evaluate on Test Set
# -------------------------------
test_loss, test_acc = model.evaluate(ds_test)
print(f"🎯 Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")

# -------------------------------
# 7. (Optional) Confusion Matrix
# -------------------------------
print("📊 Generating confusion matrix...")

y_true, y_pred = [], []
for images, labels in ds_test:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[chr(i+65) for i in range(26)])
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
