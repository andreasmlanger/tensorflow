"""
Converts TF model to TF Lite and saves in same directory
Path needs to lead to directory with 'saved_model.pb' file
"""

import tensorflow as tf
import os
import pathlib

# Input path to 'saved_model.pb' file
model_path = input('\033[94m' + 'Enter path to TensorFlow model:\n')

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
tflite_model = converter.convert()
tflite_model_file = pathlib.Path(os.path.join(model_path, 'model.tflite'))
tflite_model_file.write_bytes(tflite_model)

print('\033[36m' + 'Model successfully converted!')
