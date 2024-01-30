"""
Example script how to load and use a tflite model (works for example with 'fibonacci.tflite')
"""

import numpy as np
import tensorflow as tf
import os
import sys

PATH = input('\033[94m' + 'Enter path to directory in which tflite model is saved:\n')

files = [file for file in os.listdir(PATH) if file.endswith('.tflite')]
if len(files) == 0:
    sys.exit('No tflite model found!')
elif len(files) > 1:
    sys.exit('More than one tflite model found!')

print('\033[93m' + files[0] + ' \033[94m' + 'was found')

model_path = os.path.join(PATH, files[0])  # path to tflite model

while True:
    try:
        n = float(input('Enter a number: '))
    except ValueError:
        print('No valid number!')
        continue

    x_test = np.array([[n]], dtype=np.float32)  # X_test input for model

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, x_test)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_index)[0][0]
    print('Prediction:', prediction)
