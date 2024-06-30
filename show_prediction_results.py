import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os

from client.frames_capturer import TFLiteModel
from holistics.landmarks_extraction import load_json_file
from training.const import WEIGHTSPATH
from training.model import get_model

ROWS_PER_FRAME = 543  # number of landmarks per frame
PARQUETS_PATH = os.getenv('PARQUETSPATH')
train_df = pd.read_csv('./parquets_data.csv')

LABELS = "./labels.json"
json_file = load_json_file(LABELS)
p2s_map = {v: k for k, v in json_file.items()}  # "src/sign_to_prediction_index_map.json"
decoder = lambda x: p2s_map.get(x)

# Models to show
weights_path = [f'{WEIGHTSPATH}/lsa-9-foldall-last.h5', f'{WEIGHTSPATH}/lsa-10-foldall-last.h5',
                f'{WEIGHTSPATH}/lsa-11-foldall-last.h5']
models = [get_model() for _ in weights_path]
tflite_keras_model = TFLiteModel(islr_models=models)


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(PARQUETS_PATH + pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


# Test for just one video
# ROW = 1
# video = load_relevant_data_subset(train_df.path[ROW])
# demo_output = tflite_keras_model(video)["outputs"]
# print(f'RESULT= {decoder(str(np.argmax(demo_output.numpy(), axis=-1)))}')


def create_prediction_results(file_name):
    """
    Save the results of the prediction in a csv file
    """
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prediction", "original sign", "result"])

        for i in range(len(train_df)):
            video = load_relevant_data_subset(train_df.path[i])
            demo_output = tflite_keras_model(video)["outputs"]
            result = decoder(str(np.argmax(demo_output.numpy(), axis=-1)))
            result = result if result else "None"
            binary_result = "CORRECT" if result == train_df.sign[i] else "WRONG"
            writer.writerow([result, train_df.sign[i], binary_result])


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


create_prediction_results("prediction_results.csv")
# show_results_pie("prediction_results.csv")
