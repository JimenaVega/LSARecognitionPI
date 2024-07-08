"""
File to tests weights against its own database.
Results:
- csv file:
- json file
- pie chart generated with csv file that shows the amount of correct/wrong predictions.
- stacked bar chart that shows the amount of correct and incorrect predictions per sign.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import csv
import os
import re
import mediapipe as mp
import cv2

from holistics.landmarks_extraction import load_json_file, mediapipe_detection
from holistics.landmarks_extraction import draw
from holistics.landmarks_extraction import extract_coordinates
from training.const import WEIGHTSPATH
from training.data_process import Preprocess
from training.model import get_model


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
        outputs = [model(x) for model in self.islr_models]
        outputs = tf.keras.layers.Average()(outputs)[0]
        return {'outputs': outputs}


ROWS_PER_FRAME = 543  # number of landmarks per frame
PARQUETS_PATH = os.getenv('PARQUETSPATH')
train_df = pd.read_csv(f'{PARQUETS_PATH}/parquets_data.csv')

LABELS = "./labels.json"
json_file = load_json_file(LABELS)
p2s_map = {v: k for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
decoder = lambda x: p2s_map.get(x)

# Models to show
weights_path = [f'{WEIGHTSPATH}/lsa-10-fold0-best.h5', f'{WEIGHTSPATH}/lsa-10-fold1-best.h5',
                f'{WEIGHTSPATH}/lsa-10-fold2-best.h5']
models = [get_model() for _ in weights_path]
for model, path in zip(models, weights_path):
    model.load_weights(path)
tflite_keras_model = TFLiteModel(islr_models=models)


def get_landmarks_from_video(video_path):
    mp_holistic = mp.solutions.holistic
    sequence_data = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        cap = cv2.VideoCapture(video_path)

        # The main loop for the mediapipe detection.
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image, results = mediapipe_detection(frame, holistic)
                # draw(image, results)

                try:
                    landmarks = extract_coordinates(results)
                except:
                    landmarks = np.zeros((468 + 21 + 33 + 21, 3))
                sequence_data.append(landmarks)
            else:
                break
        cap.release()
    return sequence_data


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(PARQUETS_PATH + pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


# Test for just one video
ROW = 0

# regex = r"\d+"
# matches = re.findall(regex, train_df.path[ROW])[0]
# mp4_file = f'{matches[1:4]}_{matches[4:7]}_{matches[7:]}.mp4'
#

# path = os.getenv('CLIPPATH') + '001_001_001.mp4'
# sequence = np.array(get_landmarks_from_video(path), dtype=np.float32)  # Reads from mp4 video
# demo_output = tflite_keras_model(video)["outputs"]
# print(f'RESULT= {decoder(str(np.argmax(demo_output.numpy(), axis=-1)))}')


def create_prediction_results(file_name, from_videos=False):
    """
    Save the results of the prediction in a csv file
    file_name
    from_videos: if true then load from mp4 video and then convert it to landmark. If False
    then load from parquets file.
    """

    with open(file_name, "w", newline="") as csvfile:
        sign_result = {}
        writer = csv.writer(csvfile)
        writer.writerow(["prediction", "original sign", "result"])

        for i in range(len(train_df)):
            if from_videos:
                regex = r"\d+"
                matches = re.findall(regex, train_df.path[i])[0]
                mp4_file = f'{matches[1:4]}_{matches[4:7]}_{matches[7:]}.mp4'
                path = os.getenv('CLIPPATH') + mp4_file
                video = np.array(get_landmarks_from_video(path), dtype=np.float32)
            else:
                video = load_relevant_data_subset(train_df.path[i])

            demo_output = tflite_keras_model(video)["outputs"]
            result = decoder(str(np.argmax(demo_output.numpy(), axis=-1)))
            result = result if result else "None"
            binary_result = "CORRECT" if result == train_df.sign[i] else "WRONG"
            writer.writerow([result, train_df.sign[i], binary_result])

            if not sign_result.get(train_df.sign[i]):
                sign_result[train_df.sign[i]] = {"CORRECT": 0,
                                                 "WRONG": 0}
            else:
                sign_result[train_df.sign[i]][binary_result] += 1

        return sign_result


def show_results_pie(file_name):
    with open(file_name, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        correct_count = 0
        wrong_count = 0

        for row in reader:
            result = row[2]  # Assuming the result is in the third column (index 2)
            if result == "CORRECT":
                correct_count += 1
            elif result == "WRONG":
                wrong_count += 1
            else:
                print(f"Warning: Unexpected result found: {result}")

    labels = ['correct', 'wrong']
    sizes = [correct_count / len(train_df), wrong_count / len(train_df)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


def show_stacked_bar_plot(json_file):
    with open(json_file, "r") as file:
        sign_result = json.load(file)
    signs = []
    correct = []
    wrong = []
    for sign, result in sign_result.items():
        signs.append(sign)
        correct.append(result.get("CORRECT"))
        wrong.append(result.get("WRONG"))

    sign_num = [i for i, _ in enumerate(signs)]
    plt.figure(figsize=(10, 6))
    bar_width = 0.35  # Adjust bar width for better separation
    plt.bar(sign_num, correct, bar_width, color='g', label="CORRECT")
    plt.bar(sign_num, wrong, bar_width, bottom=correct, color='r', label="WRONG")

    plt.xlabel("Signs")
    plt.ylabel("Count")
    plt.title("Stacked Bar Plot: Correct vs. Wrong Sign Recognition")

    # Show legend
    plt.legend()
    plt.show()


def save_dict_to_json(data, filename):
    """Saves a Python dictionary to a JSON file.

  Args:
      data: The dictionary to save.
      filename: The name of the file to save the data to.
  """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)  # Add indent for readability


file_name = 'prediction_results_fold012-mp3videos-2'
# file_name = 'test'
csv_name = os.getenv('DATAPATH') + f'/artificial_intelligence/training_results/csv/{file_name}.csv'
sign_result = create_prediction_results(csv_name, from_videos=True)
# json_file = os.getenv('DATAPATH')  + f'/artificial_intelligence/training_results/json/{file_name}.json'
# save_dict_to_json(sign_result, json_file)
# show_results_pie(csv_name)
# show_stacked_bar_plot(json_file)
