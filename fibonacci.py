"""
Tensorflow NN to predict the Fibonacci sequence
"""

import keras
import numpy as np


MODEL_PATH = 'E:/models/fibonacci.keras'
EPOCHS = 500


def generate_fibonacci_series(n=100):
    arr = [0, 1]
    a, b = arr[0], arr[1]
    for _ in range(n - 1):
        a, b = b, a + b
        arr.append(b)
    return np.array(arr, dtype=np.float32)


try:
    model = keras.models.load_model(MODEL_PATH)

except ValueError:
    fibonacci = generate_fibonacci_series()

    # Create normalized X and y
    X = fibonacci[:-1] / fibonacci[-1]
    y = fibonacci[1:] / fibonacci[-1]

    # Build the model
    model = keras.Sequential([
        keras.layers.Input(shape=[1]),
        keras.layers.Dense(units=1)
    ])
    optimizer = keras.optimizers.SGD(learning_rate=0.5)

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=EPOCHS)

    # Save the model
    model.save(MODEL_PATH)

w1, w2 = model.weights
print(f'Determined weights for the Fibonacci series: {w1[0][0]} (w1) and {w2[0]} (w2)')


def test_model(m):
    while True:
        try:
            n = float(input('Enter a number: '))
            x = np.expand_dims(np.array([n]), axis=0)
            prediction = m.predict(x)[0][0].round()
            print('Prediction of next Fibonacci number:', int(prediction))
        except ValueError:
            print('No valid number!')


test_model(model)
