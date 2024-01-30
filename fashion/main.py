"""
Tensorflow NNs to recognize fashion objects
"""

import tensorflow as tf
from tensorflow import keras
from utils import *
print(f'TF version: {tf.__version__}')


EPOCHS = 25
# NN = 'NN'
NN = 'CNN'  # convolutional neural network
MODEL_PATH = f'E:/models/tensorflow/fashion_{NN}'


# Load training and test data
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(f'Shape of images: {train_images.shape} | {test_images.shape}')
print(f'Shape of labels: {train_labels.shape} | {test_labels.shape}')

show_random_image(train_images, train_labels)
show_first_25_images(train_images, train_labels)

# Reshape images for CNN
if NN == 'CNN':
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

# Normalize pixel values
train_images = np.divide(train_images, 255.0)
test_images = np.divide(test_images, 255.0)

try:
    model = keras.models.load_model(MODEL_PATH)
    model.summary()  # Check architecture

except OSError:
    # Build the model
    if NN == 'CNN':
        model = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
    else:
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
            ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=EPOCHS)

    # Save the model
    model.save(MODEL_PATH)


# Evaluate model
model.evaluate(test_images, test_labels)

show_predictions(model, test_images, test_labels)
