import cv2
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
import base64
from datetime import datetime
import io
import os
import time
import urllib.error
import urllib.request
from model import face_encoder
from send_email import send_email_notification


MIN_RECORDING_TIME = 5  # minimum time that recording runs after face has been detected
THRESHOLD = 0.5  # similarity threshold for face recognition


def resource_path(relative_path):
    return str(os.path.join(os.path.dirname(__file__), relative_path))


def get_encodings_for_known_faces():
    face_dir = resource_path('data/images')
    d = {}
    for image_path in os.listdir(face_dir):
        img = load_image_from_file(os.path.join(face_dir, image_path))
        img = preprocess_image(img)
        embedding = get_face_embedding(img)
        name = image_path.split('.')[0]
        d[name] = embedding
    return d


def load_image_from_file(image_path):
    img = cv2.imread(image_path)
    return img


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_LINEAR)  # resize to 160x160 pixels
    image = image / 255.0  # normalize pixel values to be between 0 and 1
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # convert to TF tensor
    return image


def get_face_embedding(image):
    embedding = face_encoder.predict(np.expand_dims(image, axis=0))[0]
    return embedding


def calculate_similarity(x, y):
    return cosine(x, y)


def frame_to_image(frame):
    _, buffer = cv2.imencode('.png', frame)
    io_buf = io.BytesIO(buffer)
    image = base64.b64encode(io_buf.read()).decode()
    return image


def get_byte_response_from_mobile(ip):
    url = f'http://{ip}:8080/shot.jpg'  # URL for mobile IP webcam
    while True:
        try:
            with urllib.request.urlopen(url) as r:
                response = r.read()
                if response:
                    return response
        except urllib.error.URLError:
            print('No IP Webcam connected!')
        except Exception as ex:
            print(ex)  # Windows error when connection is forcefully closed


def get_frame_from_mobile(ip):
    response = get_byte_response_from_mobile(ip)
    img = np.asarray(bytearray(response), dtype=np.uint8)
    frame = cv2.imdecode(img, -1)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame


def get_face_name(face_image, face_recognition):
    if face_recognition:
        processed_image = preprocess_image(face_image)
        embedding = get_face_embedding(processed_image)
        for face in KNOWN_FACES:
            similarity = calculate_similarity(embedding, KNOWN_FACES[face])
            if similarity < THRESHOLD:
                return face, (139, 153, 27)  # green
    return 'Unknown', (71, 40, 196)  # red


def label_face(frame, face_coordinates, face_recognition=True):
    left, top, w, h = face_coordinates
    right = left + w
    bottom = top + h
    face_image = frame[top:bottom, left:right]

    # Run face recognition if enabled to obtain name
    name, color = get_face_name(face_image, face_recognition)

    # Draw rectangle around face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Put label with name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


def record_video(frame, faces, record_until, out, email=False):
    now = time.time()
    if len(faces) > 0:
        if now > record_until:  # no current recording
            out = start_recording(frame, email)
        record_until = now + MIN_RECORDING_TIME  # update last time of seen face or body
    if now < record_until:
        out.write(frame)
    elif out:
        out.release()
        out = None
        print('Recording ended!')
    return record_until, out


def start_recording(frame, email):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    recordings_dir = 'recordings'
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
    video_path = os.path.join(recordings_dir, f'{current_time}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # define the 4 character video code for recording output
    fps = 20
    frame_size = (frame.shape[1], frame.shape[0])
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    if email:
        im = frame_to_image(frame)
        send_email_notification(im, current_time)
    print('Recording started!')
    return out


FACE_CASCADE = cv2.CascadeClassifier(resource_path('haarcascade_frontalface_default.xml'))
KNOWN_FACES = get_encodings_for_known_faces()
