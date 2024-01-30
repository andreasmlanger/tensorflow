import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QWidget, QHBoxLayout, QVBoxLayout


CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


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


def start_application(model):
    class MainWindow(QMainWindow):
        def __init__(self, m):
            super().__init__()
            self.setWindowTitle('Digits Classification')
            self.setWindowIcon(QIcon('data/brain.png'))

            self.canvas = Canvas(m)
            self.canvas.initialize_plot()

            self.button_size = QSize(40, 40)

            self.delete_button = QPushButton()
            self.delete_button.setFixedSize(self.button_size)
            self.delete_button.setIcon(QIcon(QPixmap('data/trash.png')))
            self.delete_button.setIconSize(self.button_size)
            self.delete_button.setFlat(True)
            self.delete_button.pressed.connect(self.canvas.clear_canvas)
            self.delete_button.setEnabled(False)

            self.submit_button = QPushButton()
            self.submit_button.setFixedSize(self.button_size)
            self.submit_button.setIcon(QIcon(QPixmap('data/brain.png')))
            self.submit_button.setIconSize(self.button_size)
            self.submit_button.setFlat(True)
            self.submit_button.pressed.connect(self.canvas.analyze_number)
            self.submit_button.setEnabled(False)

            buttons = QHBoxLayout()
            buttons.addWidget(self.delete_button)
            buttons.addWidget(self.submit_button)

            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            layout.addLayout(buttons)

            widget = QWidget()
            widget.setLayout(layout)
            self.setCentralWidget(widget)

    class Canvas(QLabel):
        def __init__(self, m):
            super().__init__()
            self.pixmap = QPixmap(300, 300)
            self.pixmap.fill(QColor('white'))
            self.setPixmap(self.pixmap)
            self.model = m

            self.last_x, self.last_y = None, None
            self.pen_color = QColor('#000000')

        @staticmethod
        def initialize_plot():
            plt.figure(figsize=(4, 4))
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])

        def clear_canvas(self):
            self.pixmap = QPixmap(300, 300)
            self.pixmap.fill(QColor('white'))
            self.setPixmap(self.pixmap)

            window.delete_button.setEnabled(False)
            window.submit_button.setEnabled(False)

        @staticmethod
        def crop_and_center(im):
            im_array = np.array(im)
            black_pixel_coordinates = np.argwhere(im_array == 255)  # Find coordinates of black pixels
            min_y = black_pixel_coordinates[:, 0].min()
            min_x = black_pixel_coordinates[:, 1].min()
            max_y = black_pixel_coordinates[:, 0].max()
            max_x = black_pixel_coordinates[:, 1].max()
            w, h = max_x - min_x, max_y - min_y
            im = im.crop((min_x, min_y, max_x, max_y))
            im = ImageOps.expand(im, border=int(max(w, h) * 0.2), fill=0)
            return im

        @staticmethod
        def pixmap_to_pil_bw_image(pixmap):
            image = pixmap.toImage()
            image = Image.fromqimage(image)
            image = ImageOps.invert(image)
            image = ImageOps.grayscale(image)
            return image

        def analyze_number(self):
            image = self.pixmap_to_pil_bw_image(self.pixmap)
            self.clear_canvas()
            image = self.crop_and_center(image)
            image = image.resize((28, 28), Image.LANCZOS)
            image = np.array(image.getdata()).reshape(*image.size)
            prediction = self.model.predict((np.expand_dims(image, 0)))
            predicted_label = int(np.argmax(prediction))
            if plt.get_fignums():
                plt.close('all')

            self.initialize_plot()

            plt.imshow(image)
            plt.title('Prediction: {}'.format(CLASS_NAMES[predicted_label]))
            plt.show()

        def mouseMoveEvent(self, e):
            if self.last_x is None:  # First mouse event
                self.last_x = e.position().x()
                self.last_y = e.position().y()
                return

            painter = QPainter(self.pixmap)
            p = painter.pen()
            p.setWidth(16)
            p.setColor(self.pen_color)
            painter.setPen(p)
            painter.drawLine(int(self.last_x), int(self.last_y), int(e.position().x()), int(e.position().y()))
            painter.end()
            self.update()

            # Update the origin for next time
            self.last_x = e.position().x()
            self.last_y = e.position().y()

            window.delete_button.setEnabled(True)
            window.submit_button.setEnabled(True)

            painter.end()

            self.setPixmap(self.pixmap)

        def mouseReleaseEvent(self, e):
            self.last_x = None
            self.last_y = None

    app = QApplication([])
    window = MainWindow(model)
    window.show()
    app.exec()
