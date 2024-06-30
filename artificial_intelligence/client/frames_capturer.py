"""
Real-time ASL (American Sign Language) Recognition

This script uses a pre-trained TFLite model to perform real-time ASL recognition using webcam feed. It utilizes the MediaPipe library for hand tracking and landmark extraction.

Author: 209sontung

Date: May 2023
"""
import numpy as np
import tensorflow as tf
import mediapipe as mp
from dotenv import load_dotenv
import cv2
import time
import os

from training.model import get_model
from training.data_process import Preprocess
from training.const import WEIGHTSPATH
from holistics.landmarks_extraction import mediapipe_detection
from holistics.landmarks_extraction import draw
from holistics.landmarks_extraction import extract_coordinates
from holistics.landmarks_extraction import load_json_file

load_dotenv()


SEQ_LEN = 90
THRESHOLD = 0.1
RT_CAMERA = True
CLIP_PATH = os.getenv('CLIPPATH') + '004_001_001.mp4'


class TFLiteModel(tf.Module):
    """
    TensorFlow Lite model that takes input tensors and applies:
        – A Preprocessing Model
        – The ISLR model
    """

    def __init__(self, islr_models):
        """
        Initializes the TFLiteModel with the specified preprocessing model and ISLR model.
        """
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.prep_inputs = Preprocess()
        self.islr_models = islr_models

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        """
        Applies the feature generation model and main model to the input tensors.

        Args:
            inputs: Input tensor with shape [batch_size, 543, 3].

        Returns:
            A dictionary with a single key 'outputs' and corresponding output tensor.
        """
        x = self.prep_inputs(tf.cast(inputs, dtype=tf.float32))

        # padding_size = (384 - x.shape[1]) // 2

        # padding_begin = tf.fill([1, padding_size, 708], -100.0) if (384 - x.shape[1]) % 2 == 0 else tf.fill([1, padding_size+1, 708], -100.0)
        
        # padding_end = tf.fill([1, padding_size, 708], -100.0)

        # padded_x = tf.concat([padding_begin, x, padding_end], axis=1)

        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

LABELS = "./labels.json"

json_file = load_json_file(LABELS)
s2p_map = {k.lower(): v for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
p2s_map = {v: k for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

weights_path = [f'{WEIGHTSPATH}/lsa-9-foldall-last.h5', f'{WEIGHTSPATH}/lsa-10-foldall-last.h5', f'{WEIGHTSPATH}/lsa-11-foldall-last.h5'] #
models = [get_model() for _ in weights_path]

# Load weights from the weights file.
for model, path in zip(models, weights_path):
    model.load_weights(path)
    # model.summary()


def real_time_asl():
    """
    Perform real-time ASL recognition using webcam feed.

    This function initializes the required objects and variables, captures frames from the webcam, processes them for hand tracking and landmark extraction, and performs ASL recognition on a sequence of landmarks.

    Args:
        None

    Returns:
        None
    """
    res = []
    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []

    start = time.time()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        if RT_CAMERA:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(CLIP_PATH)

        # The main loop for the mediapipe detection.
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]

                # padding_size = (384 - prediction.shape[1]) // 2

                # padding_begin = tf.fill([1, padding_size, 708], -100.0) if (384 - prediction.shape[1]) % 2 == 0 else tf.fill([1, padding_size+1, 708], -100.0)
                
                # padding_end = tf.fill([1, padding_size, 708], -100.0)

                # padded_x = tf.concat([padding_begin, prediction, padding_end], axis=1)

                # if np.max(prediction.numpy(), axis=-1) > THRESHOLD:
                sign = np.argmax(prediction.numpy(), axis=-1)

                print(f'Prediction: {p2s_map[str(sign)]} with {round((np.max(prediction.numpy(), axis=-1)) * 100, 1)}% accuracy')

                break

            start = time.time()

            image, results = mediapipe_detection(frame, holistic)
            draw(image, results)

            try:
                landmarks = extract_coordinates(results)
            except:
                landmarks = np.zeros((468 + 21 + 33 + 21, 3))
            sequence_data.append(landmarks)

            sign = ""

            # Generate the prediction for the given sequence data.
            if (RT_CAMERA and len(sequence_data) % SEQ_LEN == 0) or (not RT_CAMERA and not ret):
                prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]
                sign = np.argmax(prediction.numpy(), axis=-1)
                sequence_data = []

            image = cv2.flip(image, 1)

            cv2.putText(image, f"{len(sequence_data)}", (3, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            image = cv2.flip(image, 1)

            # Insert the sign in the result set if sign is not empty.
            if sign != "" and decoder(str(sign)) not in res:
                res.insert(0, p2s_map[str(sign)])

            # Get the height and width of the image
            height, width = image.shape[0], image.shape[1]

            # Create a white column
            white_column = np.ones((height // 8, width, 3), dtype='uint8') * 255

            # Flip the image vertically
            image = cv2.flip(image, 1)

            # Concatenate the white column to the image
            image = np.concatenate((white_column, image), axis=0)

            cv2.putText(image, f"{', '.join(str(x) for x in res)}", (3, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Webcam Feed', image)

            # Wait for a key to be pressed.
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


real_time_asl()
