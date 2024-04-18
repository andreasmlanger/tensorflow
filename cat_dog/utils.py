import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.utils import img_to_array, load_img


def make_predictions(model, size, test_dir, class_names):
    while True:
        random_img = random.choice(os.listdir(test_dir))
        img_path = str(os.path.join(test_dir, random_img))
        img = load_img(img_path, target_size=(size, size))
        x = img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, 0)
        image_tensor = np.vstack([x])
        prediction = model.predict(image_tensor)[0][0]

        plt.figure(figsize=(4, 4))
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(mpimg.imread(img_path))

        label = class_names[0] if prediction < 0.5 else class_names[1]
        probability = 1 - prediction if prediction < 0.5 else prediction

        plt.title(f'This is a {label} ({round(100 * probability, 1)}%)')
        plt.show()
