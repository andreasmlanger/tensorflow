import random
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


def show_random_image(images, labels):
    plt.figure()
    idx = random.randint(0, images.shape[0])
    plt.imshow(images[idx])
    plt.colorbar()
    plt.title(CLASS_NAMES[labels[idx]])
    plt.grid(False)
    plt.show()


def show_first_25_images(images, labels):
    plt.figure(figsize=(7, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(CLASS_NAMES[labels[i]])

    plt.tight_layout()
    plt.show()


def show_predictions(model, images, labels):
    while True:
        idx = random.randint(0, images.shape[0] - 1)
        img = images[idx]
        prediction = model.predict((np.expand_dims(img, 0)))

        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img)

        predicted_label = int(np.argmax(prediction))

        plt.title('{} ({:2.0f}%)'.format(CLASS_NAMES[predicted_label], 100 * np.max(prediction)))
        plt.xlabel('Actual Item: {}'.format(CLASS_NAMES[labels[idx]]))

        plt.subplot(1, 2, 2)

        plt.grid(False)
        plt.xticks(range(10), CLASS_NAMES, rotation=45)
        plt.yticks([])
        plt.ylim([0, 1])

        this_plot = plt.bar(range(10), prediction[0], color='darkslategray')
        this_plot[predicted_label].set_color('crimson')
        this_plot[labels[idx]].set_color('steelblue')

        plt.tight_layout()

        plt.show()
