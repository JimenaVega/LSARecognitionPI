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
import os

from ..artificial_intelligence.training.model import get_model
from ..artificial_intelligence.training.data_process import Preprocess
from ..artificial_intelligence.training.const import WEIGHTSPATH
from holistics.landmarks_extraction import mediapipe_detection
from holistics.landmarks_extraction import extract_coordinates
from holistics.landmarks_extraction import load_json_file


load_dotenv()


class TFLiteModel(tf.Module):

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
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATAPATH = os.getenv('DATAPATH')
LABELS = f"{DATAPATH}labels.json"

json_file = load_json_file(LABELS)
s2p_map = {k.lower(): v for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
p2s_map = {v: k for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
encoder = lambda x: s2p_map.get(x.lower())
decoder = lambda x: p2s_map.get(x)

weights_path = [f'{WEIGHTSPATH}/lsa-raw-nan-seed44-fold1-best.h5']
models = [get_model() for _ in weights_path]

# Load weights from the weights file.
for model, path in zip(models, weights_path):
    model.load_weights(path)
    # model.summary()


def process_sign(file):

    tflite_keras_model = TFLiteModel(islr_models=models)
    sequence_data = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


        cap = cv2.VideoCapture(file)

        # The main loop for the mediapipe detection.
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                image, results = mediapipe_detection(frame, holistic)

                try:
                    landmarks = extract_coordinates(results)
                except:
                    landmarks = np.zeros((468 + 21 + 33 + 21, 3))
                sequence_data.append(landmarks)

            # Generate the prediction for the given sequence data.
            else:
                prediction = tflite_keras_model(np.array(sequence_data, dtype=np.float32))["outputs"]
                sign = np.argmax(prediction.numpy(), axis=-1)

        cap.release()

        return sign
