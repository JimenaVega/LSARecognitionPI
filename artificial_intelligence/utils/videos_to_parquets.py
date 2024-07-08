import os
import cv2
import json
import csv
import datetime
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import product
from scipy.spatial import distance
from dotenv import load_dotenv
from holistics.landmarks_extraction import mediapipe_detection, extract_coordinates

load_dotenv()

HAND_FLATTEN_POINTS = 42
POSE_FLATTEN_POINTS = 46

# TO-FIX: cargar estos labels del labels.json
signs_codes = {
    '1': 'Opaque',
    '2': 'Red',
    '3': 'Green',
    '4': 'Yellow',
    '5': 'Bright',
    '6': 'Light-blue',
    '7': 'Colors',
    '8': 'Pink',
    '9': 'Women',
    '10': 'Enemy',
    '11': 'Son',
    '12': 'Man',
    '13': 'Away',
    '14': 'Drawer',
    '15': 'Born',
    '16': 'Learn',
    '17': 'Call',
    '18': 'Skimmer',
    '19': 'Bitter',
    '20': 'Sweet milk',
    '21': 'Milk',
    '22': 'Water',
    '23': 'Food',
    '24': 'Argentina',
    '25': 'Uruguay',
    '26': 'Country',
    '27': 'Last name',
    '28': 'Where',
    '29': 'Mock',
    '30': 'Birthday',
    '31': 'Breakfast',
    '32': 'Photo',
    '33': 'Hungry',
    '34': 'Map',
    '35': 'Coin',
    '36': 'Music',
    '37': 'Ship',
    '38': 'None',
    '39': 'Name',
    '40': 'Patience',
    '41': 'Perfume',
    '42': 'Deaf',
    '43': 'Trap',
    '44': 'Rice',
    '45': 'Barbecue',
    '46': 'Candy',
    '47': 'Chewing-gum',
    '48': 'Spaghetti',
    '49': 'Yogurt',
    '50': 'Accept',
    '51': 'Thanks',
    '52': 'Shut down',
    '53': 'Appear',
    '54': 'To land',
    '55': 'Catch',
    '56': 'Help',
    '57': 'Dance',
    '58': 'Bathe',
    '59': 'Buy',
    '60': 'Copy',
    '61': 'Run',
    '62': 'Realize',
    '63': 'Give',
    '64': 'Find',
}


def get_lsa64_metadata(dataset_version='raw'):
    clips_data = {}

    folder_path = rf"C:\Users\alejo\Downloads\lsa64_{dataset_version}\all"

    files = os.listdir(folder_path)

    for file_name in files:
        clip = cv2.VideoCapture(f'{folder_path}\{file_name}')

        if not clip.isOpened():
            print(f'Couldnt open {file_name}')
            continue

        sign_code = int(file_name.split('_')[0])
        sign = signs_codes[str(sign_code)]
        signer = int(file_name.split('_')[1])
        secuence = int(file_name.split('_')[2].split('.')[0])

        print(f'sign:{sign}    signer:{signer}     secuence:{secuence}')

        if sign not in clips_data.keys():
            clips_data[sign] = {}

        clips_data[sign][str((signer - 1) * 5 + secuence)] = {'fps': clip.get(cv2.CAP_PROP_FPS),
                                                              'frames': int(clip.get(cv2.CAP_PROP_FRAME_COUNT))}

        clip.release()

    with open(f'lsa64_{dataset_version}_metadata.json', "w") as outfile:
        json.dump(clips_data, outfile, indent=4)


def plot_sign_metadata(sign, dataset_version='raw'):
    json_file = open(f'lsa64_{dataset_version}_metadata.json')
    metadata = json.load(json_file)
    json_file.close()

    if not sign in metadata.keys():
        return

    sign_metadata = metadata[sign]

    tags = list(sign_metadata.keys())

    fps = [data['fps'] for data in sign_metadata.values()]
    avg_fps = sum(fps) / len(fps)

    frames = [data['frames'] for data in sign_metadata.values()]
    avg_frames = sum(frames) / len(frames)

    fig, axs = plt.subplots(2, 1, figsize=(15, 6))

    fps_bars = axs[0].bar(tags, fps, color='skyblue')
    axs[0].set_title(f'Clips - FPS ({sign}) | [{dataset_version} version dataset]')
    axs[0].set_xlabel("Clip")
    axs[0].set_ylabel("FPS")
    axs[0].set_ylim(bottom=0, top=max(fps) + 15)

    avg_fps_line = axs[0].axhline(y=avg_fps, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1,
                                  label=f'Avg. FPS: {avg_fps:.2f}')
    axs[0].legend(handles=[avg_fps_line])

    for bar, value in zip(fps_bars, fps):
        bar_height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar_height + 0.05
        tag = axs[0].text(x_pos, y_pos, value, ha="center", va="bottom", fontsize=8, rotation=45)

    frames_bar = axs[1].bar(tags, frames, color='lightgreen')
    axs[1].set_title(f'Clips - Frames ({sign}) | [{dataset_version} version dataset]')
    axs[1].set_xlabel("Clip")
    axs[1].set_ylabel("Frames")
    axs[1].set_ylim(bottom=0, top=max(frames) + 20)

    avg_frames_line = axs[1].axhline(y=avg_frames, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1,
                                     label=f'Avg. Frames: {avg_frames:.0f}')
    axs[1].legend(handles=[avg_frames_line])

    for bar, value in zip(frames_bar, frames):
        bar_height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar_height + 0.05
        tag = axs[1].text(x_pos, y_pos, value, ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout()
    plt.show()


def plot_lsa64_metadata(dataset_version='raw'):
    json_file = open(f'lsa64_{dataset_version}_metadata.json')
    metadata = json.load(json_file)
    json_file.close()

    tags = list(metadata.keys())

    fps = []
    frames = []

    for sign in tags:
        sign_fps = [data['fps'] for data in metadata[sign].values()]
        sign_frames = [data['frames'] for data in metadata[sign].values()]

        fps.append(round(sum(sign_fps) / len(sign_fps), 2))
        frames.append(int(sum(sign_frames) / len(sign_frames)))

    avg_fps = sum(fps) / len(fps)
    avg_frames = sum(frames) / len(frames)

    fig, axs = plt.subplots(2, 1, figsize=(15, 6))

    fps_bars = axs[0].bar(tags, fps, color='skyblue')
    axs[0].set_title(f'Signs - Average FPS | [{dataset_version} version dataset]')
    axs[0].set_xlabel("Sign")
    axs[0].set_ylabel("Average FPS")
    axs[0].set_ylim(bottom=0, top=max(fps) + 15)

    avg_fps_line = axs[0].axhline(y=avg_fps, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1,
                                  label=f'Total avg. FPS: {avg_fps:.2f}')
    axs[0].legend(handles=[avg_fps_line])

    for bar, value in zip(fps_bars, fps):
        bar_height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar_height + 0.05
        axs[0].text(x_pos, y_pos, value, ha="center", va="bottom", fontsize=8, rotation=45)

    frames_bar = axs[1].bar(tags, frames, color='lightgreen')
    axs[1].set_title(f'Signs - Average frames | [{dataset_version} version dataset]')
    axs[1].set_xlabel("Sign")
    axs[1].set_ylabel("Average frames")
    axs[1].set_ylim(bottom=0, top=max(frames) + 20)

    avg_frames_line = axs[1].axhline(y=avg_frames, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1,
                                     label=f'Total avg. frames: {avg_frames:.0f}')
    axs[1].legend(handles=[avg_frames_line])

    for bar, value in zip(frames_bar, frames):
        bar_height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar_height + 0.05
        axs[1].text(x_pos, y_pos, value, ha="center", va="bottom", fontsize=8, rotation=45)

    for ax in axs.flat:  # Iterate over flattened version of axs
        ax.tick_params(axis='x', rotation=70)

    plt.tight_layout()
    plt.show()


def add_dataset_padding(dataset_version='raw', sequences=50):
    # Define the actions (signs) that will be recorded and stored in the dataset
    PATH = os.path.join(f'data_{dataset_version}_zero_padded')

    actions = list(signs_codes.values())

    # Create directories for each action, sequence, and frame in the dataset
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence), 'hands'))
            os.makedirs(os.path.join(PATH, action, str(sequence), 'pose'))
        except:
            pass

    signs_dirs = os.listdir(f'data_{dataset_version}')

    longest_frames = 0

    for sign_dir in signs_dirs:
        sequence_dirs = os.listdir(f'data_{dataset_version}/{sign_dir}')

        for sequence_dir in sequence_dirs:
            frames = os.listdir(f'data_{dataset_version}/{sign_dir}/{sequence_dir}/hands')

            if len(frames) > longest_frames:
                longest_frames = len(frames)

    for sign_dir in signs_dirs:
        sequence_dirs = os.listdir(f'data_{dataset_version}/{sign_dir}')

        for sequence_dir in sequence_dirs:
            hands_frames = os.listdir(f'data_{dataset_version}/{sign_dir}/{sequence_dir}/hands')
            hands_padded_frames = []

            pose_frames = os.listdir(f'data_{dataset_version}/{sign_dir}/{sequence_dir}/pose')
            pose_padded_frames = []

            total_padding = longest_frames - len(hands_frames)

            if total_padding % 2 == 0:
                lower_padding = int(total_padding / 2)
                upper_padding = lower_padding
            else:
                lower_padding = int(total_padding / 2)
                upper_padding = lower_padding + 1

            for i in range(lower_padding):
                hands_padded_frames.append(np.zeros(HAND_FLATTEN_POINTS))
                pose_padded_frames.append(np.zeros(HAND_FLATTEN_POINTS))

            for frame in hands_frames:
                hands_padded_frames.append(np.load(f'data_{dataset_version}/{sign_dir}/{sequence_dir}/hands/{frame}'))

            for frame in pose_frames:
                pose_padded_frames.append(np.load(f'data_{dataset_version}/{sign_dir}/{sequence_dir}/pose/{frame}'))

            for i in range(upper_padding):
                hands_padded_frames.append(np.zeros(HAND_FLATTEN_POINTS))
                pose_padded_frames.append(np.zeros(HAND_FLATTEN_POINTS))

            for i in range(len(hands_padded_frames)):
                frame_path = os.path.join(PATH, sign_dir, str(sequence_dir), 'hands', str(i))
                np.save(frame_path, hands_padded_frames[i])

                frame_path = os.path.join(PATH, sign_dir, str(sequence_dir), 'pose', str(i))
                np.save(frame_path, pose_padded_frames[i])

            print(f'Padded for {sign_dir} {sequence_dir}/50 completed.')


def create_dataset(dataset_version='raw', sequences=50):
    # Define the actions (signs) that will be recorded and stored in the dataset
    PATH = os.path.join(f'data_{dataset_version}')

    actions = list(signs_codes.values())

    # Create directories for each action, sequence, and frame in the dataset
    for action, sequence in product(actions, range(sequences)):
        try:
            os.makedirs(os.path.join(PATH, action, str(sequence), 'hands'))
            os.makedirs(os.path.join(PATH, action, str(sequence), 'pose'))
        except:
            pass

    folder_path = rf"C:\Users\alejo\Downloads\lsa64_{dataset_version}\all"

    files = os.listdir(folder_path)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:

        for file_name in files:
            clip = cv2.VideoCapture(f'{folder_path}\{file_name}')

            if not clip.isOpened():
                print(f'Couldnt open {file_name}')
                continue

            sign_code = int(file_name.split('_')[0])
            sign = signs_codes[str(sign_code)]
            signer = int(file_name.split('_')[1])
            sequence = (signer - 1) * 5 + int(file_name.split('_')[2].split('.')[0]) - 1

            clips_frames = int(clip.get(cv2.CAP_PROP_FRAME_COUNT))

            for i in range(clips_frames):
                _, frame = clip.read()

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference
                # frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame using the model
                results = holistic.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # We take the shoulders, and nose points
                # The nose point is used for translation, and the shoulders for normalization
                left_shoulder = results.pose_landmarks.landmark[11]
                right_shoulder = results.pose_landmarks.landmark[12]
                shoulders_distance = distance.euclidean((left_shoulder.x, left_shoulder.y, left_shoulder.z),
                                                        (right_shoulder.x, right_shoulder.y, right_shoulder.z))

                nose = results.pose_landmarks.landmark[0]

                # Extract the keypoints for the left hand if present, otherwise set to zeros
                lh = np.array([[(res.x - nose.x) / shoulders_distance, (res.y - nose.y) / shoulders_distance]
                               for res in
                               results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
                    HAND_FLATTEN_POINTS)

                # Extract the keypoints for the right hand if present, otherwise set to zeros
                rh = np.array([[(res.x - nose.x) / shoulders_distance, (res.y - nose.y) / shoulders_distance]
                               for res in
                               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
                    HAND_FLATTEN_POINTS)

                # Concatenate the keypoints for both hands
                hands_landmarks = np.concatenate([lh, rh])

                # Extract the body keypoints if present, otherwise set to zeros
                pose_landmarks = np.array(
                    [[(res.x - nose.x) / shoulders_distance, (res.y - nose.y) / shoulders_distance]
                     for res in
                     results.pose_landmarks.landmark[0:23]]).flatten() if results.pose_landmarks else np.zeros(
                    POSE_FLATTEN_POINTS)

                frame_path = os.path.join(PATH, sign, str(sequence), 'hands', str(i))
                np.save(frame_path, hands_landmarks)

                frame_path = os.path.join(PATH, sign, str(sequence), 'pose', str(i))
                np.save(frame_path, pose_landmarks)

                # print(f'keypoints: {keypoints}')

            print(f'sign:{sign} ({sign_code}/64)    secuence:{sequence + 1}/50 ({clips_frames} frames)')

    add_dataset_padding(dataset_version=dataset_version)


def plot_padding(sign, sequence):
    sequence_path = os.path.join(f'data_cut/{sign}/{sequence}')
    padded_sequence_path = os.path.join(f'data_cut_zero_padded/{sign}/{sequence}')

    hands_landmarks = []
    pose_landmarks = []

    hands_padded_landmarks = []
    pose_padded_landmarks = []

    for frame in os.listdir(f'{sequence_path}/hands/'):
        pass

    x_axis_len = None


def create_parquet_files():
    # Create parquets dir where parquet files will be stored
    try:
        os.mkdir('parquets')
    except Exception as error:
        print(error)

    csv_headers = ['path', 'participant_id', 'sequence_id', 'sign']
    csv_data = []

    # folder_path = rf"C:\Users\alejo\Downloads\lsa64_{dataset_version}\all"
    folder_path = os.getenv('CLIPPATH')
    files = os.listdir(folder_path)

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        file_counter = 1

        for file in files:
            file_name = file.split('.')[0]
            sign_code = int(file.split('_')[0])
            sign = signs_codes[str(sign_code)]
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

                print(f'clip {file_counter}/{len(files)} - frame {i + 1}/{clips_frames} completed')
                break

            df = pd.DataFrame(parquet_data)

            # Convert the DataFrame to an Arrow Table
            table = pa.Table.from_pandas(df)

            file_sequence = int('1' + file_name.split('_')[0] + file_name.split('_')[1] + file_name.split('_')[2])

            parquet_url = f'parquets/{file_sequence}.parquet'

            # Write the Table to a Parquet file
            pq.write_table(table, parquet_url)

            csv_data.append({'path': parquet_url,
                             'participant_id': signer,
                             'sequence_id': file_sequence,
                             'sign': sign})

            file_counter += 1
            break
    # writing to csv file
    with open('parquets_data_cut_nan.csv', 'w', newline='') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(csv_data)


dataset_version = 'cut'

# get_lsa64_metadata(dataset_version=dataset_version)
# plot_sign_metadata(dataset_version=dataset_version, sign='Map')
# plot_lsa64_metadata(dataset_version=dataset_version)

start = datetime.datetime.now()
print(f'[{start}] Start creating parquet files')

# create_dataset(dataset_version=dataset_version)
create_parquet_files()

end = datetime.datetime.now()
print(f'[{end}] Finish creating parquet files')
print(f'Elapsed time: {end - start}')

# add_dataset_padding(dataset_version=dataset_version)
# plot_padding(sign='Accept', sequence=0)
