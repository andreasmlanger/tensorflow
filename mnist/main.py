"""
Tensorflow NNs to recognize fashion objects
"""

from utils import *
import keras
import tensorflow as tf

print(f'TF version: {tf.__version__}')

# Select dataset and neural network
# dataset = 'mnist'  # MNIST
dataset = 'fashion_mnist'  # FashionMNIST

# NN = 'NN'  # normal neural network (~97.2% / ~88.5%)
NN = 'CNN'  # convolutional neural network (~98.8% / ~90.8%)

MODEL_PATH = f'E:/models/{dataset}_{NN}.keras'

EPOCHS = 25

# Load training and test data
data = keras.datasets.mnist if dataset == 'mnist' else keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(f'Shape of images: {train_images.shape} | {test_images.shape}')
print(f'Shape of labels: {train_labels.shape} | {test_labels.shape}')

class_names = get_class_names(dataset)
show_random_image(train_images, train_labels, class_names)
show_first_25_images(train_images, train_labels, class_names)

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

except ValueError:
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
            keras.layers.Flatten(),
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

if dataset == 'mnist':
    # Run the application to recognize handwritten digits
    start_application(model)
else:
    show_predictions(model, test_images, test_labels, class_names)
