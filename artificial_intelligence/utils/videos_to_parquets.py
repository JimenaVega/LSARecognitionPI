import os
import cv2
import json
import csv
import datetime
import numpy as np
import mediapipe as mp
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from holistics.landmarks_extraction import mediapipe_detection, extract_coordinates

load_dotenv()

DATAPATH = os.getenv('DATAPATH')
JSON_LABELS = os.path.join(DATAPATH, 'labels.json')
with open(JSON_LABELS) as json_file:
    signs_codes = json.load(json_file)

code_signs = {}
for key, val in signs_codes.items():
    code_signs[val] = key

CSV_FILENAME = "new_lsa.csv"


def create_parquet_files():
    """
    Takes a folder with mp4 files and creates parquet files and a csv file with the labels corresponding to them.
    The parquets file will

    """
    csv_headers = ['path', 'participant_id', 'sequence_id', 'sign']
    csv_data = []

    parquets_path = os.getenv('PARQUETS_PATH')
    # Checks that the folder parquets exists otherwise it creates it
    if not os.path.exists(parquets_path + '/parquets'):
        os.mkdir('parquets')

    folder_path = os.getenv('CLIPS_PATH')
    files = os.listdir(folder_path)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        file_counter = 1

        for file in files:
            file_name = file.split('.')[0]
            sign_code = int(file.split('_')[0])
            sign = code_signs[str(sign_code)]
            signer = int(file.split('_')[1])

            # Create a DataFrame
            parquet_data = {'frame': [],
                            'row_id': [],
                            'type': [],
                            'landmark_index': [],
                            'x': [],
                            'y': [],
                            'z': []}

            clip = cv2.VideoCapture(os.path.join(folder_path, file))

            if not clip.isOpened():
                print(f'Couldnt open {file}')
                continue

            clips_frames = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(clips_frames):
                ret, frame = clip.read()

                # if ret:
                image, results = mediapipe_detection(frame, holistic)

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame.flags.writeable = False  # Frame no longer writeable to improve performance
                # results = holistic.process(frame)  # Process the frame using the model
                # frame.flags.writeable = True
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                face_landmarks = [[res.x, res.y, res.z] for res in
                                  results.face_landmarks.landmark] if results.face_landmarks else np.zeros(
                    (468, 3)) * np.nan
                pose_landmarks = [[res.x, res.y, res.z] for res in
                                  results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros(
                    (33, 3)) * np.nan
                lh_landmarks = [[res.x, res.y, res.z] for res in
                                results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros(
                    (21, 3)) * np.nan
                rh_landmarks = [[res.x, res.y, res.z] for res in
                                results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros(
                    (21, 3)) * np.nan

                # rh_landmarks = [[res.x, res.y, res.z] for res in
                #                 results.right_hand_landmarks.landmark] if results.right_hand_landmarks else np.zeros(
                #     shape=(21, 3))
                # lh_landmarks = [[res.x, res.y, res.z] for res in
                #                 results.left_hand_landmarks.landmark] if results.left_hand_landmarks else np.zeros(
                #     shape=(21, 3))
                # pose_landmarks = [[res.x, res.y, res.z] for res in
                #                   results.pose_landmarks.landmark] if results.pose_landmarks else np.zeros(
                #     shape=(33, 3))
                # face_landmarks = [[res.x, res.y, res.z] for res in
                #                   results.face_landmarks.landmark] if results.face_landmarks else np.zeros(
                #     shape=(468, 3))

                for lm_index in range(len(rh_landmarks)):
                    parquet_data['frame'].append(i)
                    parquet_data['row_id'].append(f'{i}-right_hand-{lm_index}')
                    parquet_data['type'].append('right_hand')
                    parquet_data['landmark_index'].append(lm_index)
                    parquet_data['x'].append(rh_landmarks[lm_index][0])
                    parquet_data['y'].append(rh_landmarks[lm_index][1])
                    parquet_data['z'].append(rh_landmarks[lm_index][2])

                for lm_index in range(len(lh_landmarks)):
                    parquet_data['frame'].append(i)
                    parquet_data['row_id'].append(f'{i}-left_hand-{lm_index}')
                    parquet_data['type'].append('left_hand')
                    parquet_data['landmark_index'].append(lm_index)
                    parquet_data['x'].append(lh_landmarks[lm_index][0])
                    parquet_data['y'].append(lh_landmarks[lm_index][1])
                    parquet_data['z'].append(lh_landmarks[lm_index][2])

                for lm_index in range(len(pose_landmarks)):
                    parquet_data['frame'].append(i)
                    parquet_data['row_id'].append(f'{i}-pose-{lm_index}')
                    parquet_data['type'].append('pose')
                    parquet_data['landmark_index'].append(lm_index)
                    parquet_data['x'].append(pose_landmarks[lm_index][0])
                    parquet_data['y'].append(pose_landmarks[lm_index][1])
                    parquet_data['z'].append(pose_landmarks[lm_index][2])

                for lm_index in range(len(face_landmarks)):
                    parquet_data['frame'].append(i)
                    parquet_data['row_id'].append(f'{i}-face-{lm_index}')
                    parquet_data['type'].append('face')
                    parquet_data['landmark_index'].append(lm_index)
                    parquet_data['x'].append(face_landmarks[lm_index][0])
                    parquet_data['y'].append(face_landmarks[lm_index][1])
                    parquet_data['z'].append(face_landmarks[lm_index][2])

                # print(f'clip {file_counter}/{len(files)} - frame {i + 1}/{clips_frames} completed')
            print(f'clip {file_counter} completed')
            df = pd.DataFrame(parquet_data)

            # Convert the DataFrame to an Arrow Table
            table = pa.Table.from_pandas(df)

            file_sequence = int('1' + file_name.split('_')[0] + file_name.split('_')[1] + file_name.split('_')[2])

            parquet_url = f'parquets/{file_sequence}.parquet'

            # Write the Table to a Parquet file
            pq.write_table(table, parquets_path + parquet_url)

            csv_data.append({'path': parquet_url,
                             'participant_id': signer,
                             'sequence_id': file_sequence,
                             'sign': sign})

            file_counter += 1

    # writing to csv file
    with open(DATAPATH + CSV_FILENAME, 'w', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(csv_data)


start = datetime.datetime.now()
print(f'[{start}] Start creating parquet files')

create_parquet_files()

end = datetime.datetime.now()
print(f'[{end}] Finish creating parquet files')
print(f'Elapsed time: {end - start}')
