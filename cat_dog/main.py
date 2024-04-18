"""
CNN and transfer learning CNN to classify cat and dog images
For training, download images from https://www.kaggle.com/c/dogs-vs-cats
Move images into 'cat' and 'dog' folders in 'training' and 'validation' directories
"""

from utils import *
import keras
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf

print(f'TF version: {tf.__version__}')

DATASET = 'cat_dog'  # cat & dog images

# NN = 'CNN'  # convolutional neural network (~91.0%)
NN = 'tCNN'  # convolutional neural network using transfer learning (~98.9%)

EPOCHS = 20 if NN == 'CNN' else 3
SIZE = 128 if NN == 'CNN' else 224  # image size in pixels

# Directories for training, validation and test data
BASE_DIR = f'E:/images/{DATASET}'
TRAINING_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test')

MODEL_PATH = f'E:/models/{DATASET}_{NN}.keras'

# Create validation data generator
validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255
).flow_from_directory(
    VALIDATION_DIR,
    target_size=(SIZE, SIZE),
    class_mode='binary'
)
class_names = list(validation_generator.class_indices)

try:
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    model.summary()  # Check architecture

    # Evaluate the model (optional)
    scores = model.evaluate(validation_generator)
    print('Accuracy: %.2f%%' % (scores[1] * 100))

except ValueError:
    # Create training data generator
    training_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    ).flow_from_directory(
        TRAINING_DIR,
        target_size=(SIZE, SIZE),
        class_mode='binary'
    )

    # Build the model
    if NN == 'tCNN':
        base = InceptionV3(input_shape=(SIZE, SIZE, 3), include_top=False, weights='imagenet')
        for layer in base.layers:
            layer.trainable = False  # freeze entire network
        last_layer = base.get_layer('mixed7')  # crop pretrained model here
        x = last_layer.output
        x = keras.layers.Flatten()(x)  # flatten to 1 dimension
        x = keras.layers.Dense(1024, activation='relu')(x)  # add fully connected layer with 1024 hidden units
        x = keras.layers.Dense(1, activation='sigmoid')(x)  # add final sigmoid layer for classification
        model = keras.Model(inputs=base.input, outputs=x)

    else:  # CNN
        model = keras.models.Sequential([
            keras.layers.InputLayer(shape=(SIZE, SIZE, 3)),
            keras.layers.Conv2D(16, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

    # Compile the model
    model.summary()  # Check architecture
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    model.fit(
        training_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    model.save(MODEL_PATH)

# Test the model on unseen test data
make_predictions(model, size=SIZE, test_dir=TEST_DIR, class_names=class_names)
