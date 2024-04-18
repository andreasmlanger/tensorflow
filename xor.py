"""
Tensorflow NN for XOR: 1 if one value is < 0.5 and the other > 0.5, else 0
"""

import keras
import numpy as np


MODEL_PATH = 'E:/models/xor.keras'
EPOCHS = 10


def generate_random_xor_data(n=100000):
    xy = np.random.rand(n, 2)
    mask = ((xy[:, 0] < 0.5) & (xy[:, 1] < 0.5)) | ((xy[:, 0] >= 0.5) & (xy[:, 1] >= 0.5))
    labels = np.where(mask, 0, 1)
    return xy, labels


try:
    model = keras.models.load_model(MODEL_PATH)

except ValueError:
    X, y = generate_random_xor_data()

    # Build the model
    model = keras.Sequential([
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

    # Train the model
    model.fit(X, y, epochs=EPOCHS, verbose=1)

    # Save the model
    model.save(MODEL_PATH)

# Test model
while True:
    try:
        t = input('Enter (x, y) tuple, e.g. 0.3, 0.6: ')
        a, b = map(float, t.split(','))
        x = np.expand_dims(np.array([a, b]), axis=0)
        prediction = model.predict(x)[0][0]
        print('XOR prediction:', int(round(prediction, 0)))
    except ValueError:
        print('No valid tuple!')
