"""
App that uses Keras Facenet to detect faces
Labeled images of faces need to be in the 'data/images' folder
Optionally record a video and send an email upon face detection
Works with in-built webcam as well as mobile IP cam
Press 'q' to quit camera
"""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QFont
from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel, QLineEdit, QMainWindow
import sys
from utils import *

MOBILE_IP = '192.168.0.59'  # default local IP that is shown in UI


def get_geometry(*args):
    return tuple(map(int, [e * Window.s for e in list(args)]))


class Window(QMainWindow):
    app = QApplication([])
    s = app.screens()[0].logicalDotsPerInch() / 96
    app.quit()

    font = QFont()
    font.setPointSize(8)

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(*get_geometry(200, 200, 260, 230))
        self.setFixedSize(self.size())
        self.setWindowTitle('Security Camera')
        self.setWindowIcon(QIcon(resource_path('data/icon.png')))

        self.start = QAction('Start', self)
        self.start.triggered.connect(self.start_watching)

        self.stop = QAction('Stop', self)
        self.stop.triggered.connect(self.stop_watching)

        close = QAction('Quit', self)
        close.setShortcut('Ctrl+Q')
        close.triggered.connect(self.close_application)

        menu = self.menuBar()

        file = menu.addMenu('File')
        file.addAction(close)

        self.startIcon = QAction(QIcon(resource_path('data/start.png')), 'Start', self)
        self.startIcon.triggered.connect(self.start_watching)

        self.stopIcon = QAction(QIcon(resource_path('data/stop.png')), 'Stop', self)
        self.stopIcon.triggered.connect(self.stop_watching)
        self.stopIcon.setEnabled(False)

        self.toolBar = self.addToolBar('toolbar')
        self.toolBar.addAction(self.startIcon)
        self.toolBar.addAction(self.stopIcon)

        self.checkbox_show = QCheckBox(self)
        self.checkbox_show.setGeometry(*get_geometry(20, 70, 30, 30))

        self.label_show = QLabel(self)
        self.label_show.setGeometry(*get_geometry(40, 70, 200, 30))
        self.label_show.setText('Show camera window')

        self.checkbox_face = QCheckBox(self)
        self.checkbox_face.setGeometry(*get_geometry(20, 95, 30, 30))

        self.label_face = QLabel(self)
        self.label_face.setGeometry(*get_geometry(40, 95, 200, 30))
        self.label_face.setText('Turn on face recognition')

        self.checkbox_recording = QCheckBox(self)
        self.checkbox_recording.setGeometry(*get_geometry(20, 120, 30, 30))

        self.label_recording = QLabel(self)
        self.label_recording.setGeometry(*get_geometry(40, 120, 200, 30))
        self.label_recording.setText('Save recorded videos')

        self.checkbox_email = QCheckBox(self)
        self.checkbox_email.setGeometry(*get_geometry(20, 145, 30, 30))

        self.label_email = QLabel(self)
        self.label_email.setGeometry(*get_geometry(40, 145, 200, 30))
        self.label_email.setText('Send email notification')

        self.ip_address = QLineEdit(self)
        self.ip_address.setFont(self.font)
        self.ip_address.setGeometry(*get_geometry(100, 180, 140, 30))
        self.ip_address.setEnabled(False)
        self.ip_address.setText(MOBILE_IP)

        self.checkbox_mobile = QCheckBox(self)
        self.checkbox_mobile.setGeometry(*get_geometry(20, 180, 30, 30))
        self.checkbox_mobile.toggled.connect(self.ip_address.setEnabled)

        self.label_mobile = QLabel(self)
        self.label_mobile.setGeometry(*get_geometry(40, 180, 100, 30))
        self.label_mobile.setText('Mobile')

        self.thread = QThread()

        self.show()

    def start_watching(self):
        self.checkbox_show.setEnabled(False)
        self.checkbox_face.setEnabled(False)
        self.checkbox_recording.setEnabled(False)
        self.checkbox_email.setEnabled(False)
        self.checkbox_mobile.setEnabled(False)
        self.ip_address.setEnabled(False)
        self.startIcon.setEnabled(False)
        self.stopIcon.setEnabled(True)

        show_camera = self.checkbox_show.isChecked()
        face_recognition = self.checkbox_face.isChecked()
        save_recordings = self.checkbox_recording.isChecked()
        send_emails = self.checkbox_email.isChecked()
        mobile_source = self.checkbox_mobile.isChecked()
        ip_address = self.ip_address.text()

        self.thread = Thread(show_camera, face_recognition, save_recordings, send_emails, mobile_source, ip_address)
        self.thread.q_key_pressed.connect(self.stop_watching)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()
        print('Camara started!')

    def on_finished(self):
        self.checkbox_show.setEnabled(True)
        self.checkbox_face.setEnabled(True)
        self.checkbox_recording.setEnabled(True)
        self.checkbox_email.setEnabled(True)
        self.checkbox_mobile.setEnabled(True)
        self.startIcon.setEnabled(True)
        if self.checkbox_mobile.isChecked():
            self.ip_address.setEnabled(True)
        print('Camera stopped!')

    def stop_watching(self):
        self.stopIcon.setEnabled(False)
        if self.thread.isRunning():
            self.thread.stop_thread()

    @staticmethod
    def close_application():
        app.quit()


class Thread(QThread):
    q_key_pressed = pyqtSignal()

    def __init__(self, show_camera, face_recognition, save_recordings, send_emails, mobile_source, ip_address):
        QThread.__init__(self)
        self.show_camera = show_camera
        self.face_recognition = face_recognition
        self.save_recordings = save_recordings
        self.send_emails = send_emails
        self.mobile_source = mobile_source
        self.ip_address = ip_address
        self.is_running = True
        self.cap = None if self.mobile_source else cv2.VideoCapture(0)

    def run(self):
        record_until, out = 0, None  # initialize recording parameters

        while self.is_running:
            if self.mobile_source:
                frame = get_frame_from_mobile(self.ip_address)
            else:
                _, frame = self.cap.read()

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray_frame, 1.3, 5)

            if self.save_recordings:
                record_until, out = record_video(frame, faces, record_until, out, email=self.send_emails)

            # If face is detected, display frame and perform face recognition
            for face_coordinates in faces:
                label_face(frame, face_coordinates, face_recognition=self.face_recognition)

            if self.show_camera:
                cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.q_key_pressed.emit()

    def stop_thread(self):
        self.is_running = False
        time.sleep(0.5)  # allow sufficient time for run function to complete
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().terminate()


if __name__ == '__main__':
    app = QApplication([])
    Gui = Window()
    sys.exit(app.exec())
