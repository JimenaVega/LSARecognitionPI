import os
import cv2
import json
import datetime
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product


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
        
        clips_data[sign][str((signer-1)*5 + secuence)] = {'fps': clip.get(cv2.CAP_PROP_FPS),
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
    
    avg_fps_line = axs[0].axhline(y=avg_fps, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1, label=f'Avg. FPS: {avg_fps:.2f}')
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
    
    avg_frames_line = axs[1].axhline(y=avg_frames, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1, label=f'Avg. Frames: {avg_frames:.0f}')
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
    
    avg_fps_line = axs[0].axhline(y=avg_fps, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1, label=f'Total avg. FPS: {avg_fps:.2f}')
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
    
    avg_frames_line = axs[1].axhline(y=avg_frames, xmin=0, xmax=len(tags), color='red', linestyle='--', lw=1, label=f'Total avg. frames: {avg_frames:.0f}')
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


def create_dataset(dataset_version='raw', sequences=50, frames=60):

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
        
        stop_condition = 0

        for file_name in files:
            clip = cv2.VideoCapture(f'{folder_path}\{file_name}')

            if not clip.isOpened():
                print(f'Couldnt open {file_name}')
                continue

            sign_code = int(file_name.split('_')[0])
            sign = signs_codes[str(sign_code)]
            signer = int(file_name.split('_')[1])
            sequence = (signer-1)*5 + int(file_name.split('_')[2].split('.')[0]) - 1

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

                # Extract the keypoints for the left hand if present, otherwise set to zeros
                lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(42)

                # Extract the keypoints for the right hand if present, otherwise set to zeros
                rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(42)

                # Concatenate the keypoints for both hands
                hands_landmarks = np.concatenate([lh, rh])

                #Extract the body keypoints if present, otherwise set to zeros
                pose_landmarks = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark[0:23]]).flatten() if results.pose_landmarks else np.zeros(46)

                frame_path = os.path.join(PATH, sign, str(sequence), 'hands', str(i))
                np.save(frame_path, hands_landmarks)

                frame_path = os.path.join(PATH, sign, str(sequence), 'pose', str(i))
                np.save(frame_path, pose_landmarks)

                # print(f'keypoints: {keypoints}')

            print(f'sign:{sign} ({sign_code}/64)    secuence:{sequence+1}/50 ({clips_frames} frames)')

            stop_condition += 1

            if stop_condition >= 100:
                break


dataset_version = 'cut'

# get_lsa64_metadata(dataset_version=dataset_version)
# plot_sign_metadata(dataset_version=dataset_version, sign='Map')
# plot_lsa64_metadata(dataset_version=dataset_version)
start = datetime.datetime.now()
print(f'[{start}] Start creating dataset')

create_dataset(dataset_version=dataset_version)

end = datetime.datetime.now()
print(f'[{end}] Finish creating dataset')
print(f'Elapsed time: {end - start}')
