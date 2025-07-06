import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True, with_info=True)

import tensorflow as tf
import tensorflow_datasets as tfds

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# Normalize and resize
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label - 1  # labels range from 1 to 26

ds_train = ds_train.map(normalize_img).cache().shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img).batch(64).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 letters
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds_train, epochs=5, validation_data=ds_test)


loss, acc = model.evaluate(ds_test)
print(f"Test Accuracy: {acc:.2f}")


model.save("ocr_model.h5")


