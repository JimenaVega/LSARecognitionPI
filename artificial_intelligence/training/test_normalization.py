import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_process import Preprocess
from holistics.landmarks_extraction import mediapipe_detection
from holistics.landmarks_extraction import extract_coordinates
from training.data_process import Preprocess
from training.const import POINT_LANDMARKS

mp_holistic = mp.solutions.holistic
pre_process = Preprocess()

file_path = '/home/alejo/Downloads/lsa64_cut/all_cut/001_001_001.mp4'

cap = cv2.VideoCapture(file_path)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequence_data = []
    normalized_data = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)

        try:
            landmarks = extract_coordinates(results)
        except:
            landmarks = np.zeros((468 + 21 + 33 + 21, 3))
        
        sequence_data.append(landmarks)
    
    normalized_data = pre_process(tf.cast(np.array(sequence_data, dtype=np.float32), dtype=tf.float32))

num_landmarks = len(POINT_LANDMARKS)

xy_pairs = normalized_data[:, :, :2*num_landmarks]

# Inicializar lista para todos los frames
normalized_xy = []

# Iterar sobre cada frame
for frame in range(xy_pairs.shape[1]):  # Itera sobre el número de frames
    frame_xy = xy_pairs[0, frame, :]  # Obtener todas las coordenadas (X, Y) para el frame actual
    
    # Organizar los pares (X, Y) en una lista de listas para el frame actual
    xy_list = [[frame_xy[i].numpy(), frame_xy[i + 1].numpy()] for i in range(0, len(frame_xy), 2)]
    
    # Añadir la lista de pares (X, Y) del frame actual a la lista principal
    normalized_xy.append(xy_list)

raw_data = []

# sequence_data = [list(tf.gather(frame, POINT_LANDMARKS).numpy()) for frame in sequence_data]

for frame in sequence_data:
    frame_data = []

    for landmark in tf.gather(frame, POINT_LANDMARKS).numpy():
        frame_data.append([landmark[0], landmark[1]])
    
    raw_data.append(frame_data)

class Animation:

    connections = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # Face points
                   (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # Right arm points
                   (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # Left arm points
                   (11, 12), (11, 23), (12, 24), (23, 24),  # Upper body points
                   (0, 5), (5, 8),
                   (0, 9), (9, 12),
                   (0, 13), (13, 16),
                   (0, 17), (17, 20),
                   (0, 21), (21, 24)
                        #  (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # Right leg points
                        #  (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left leg points
                        ]

    def __init__(self) -> None:
        self.landmarks = raw_data
        self.total_frames = len(raw_data)

        # Calculate global limits
        all_landmarks = np.vstack(self.landmarks)
        self.x_axis_min = np.nanmin(all_landmarks[:, 0])
        self.x_axis_max = np.nanmax(all_landmarks[:, 0])
        self.y_axis_min = np.nanmin(all_landmarks[:, 1])
        self.y_axis_max = np.nanmax(all_landmarks[:, 1])

    def run_animation(self):
        # Creates the plot
        fig, self.ax = plt.subplots()

        # The animation interval is based on the frames per second of the video
        animation_interval = 200  # miliseconds

        # Creates the animation
        animation = FuncAnimation(fig, self.update_plot, frames=self.total_frames, interval=animation_interval)

        # Show the animation
        plt.show()

    def update_plot(self, frame):
        """
        This method updates the values on the plot, making the animation
        """

        # We clean the previous plot
        self.ax.cla()

        frame_landmarks = self.landmarks[frame]
        for landmark in frame_landmarks:
            # Plots the new 2D points
            self.ax.scatter(landmark[0], landmark[1], c='r', marker='o')

        # # This draws a line between the landmarks
        # for point_pair in self.connections:
        #     self.ax.plot([frame_landmarks[point_pair[0]][0], frame_landmarks[point_pair[1]][0]],
        #                  [frame_landmarks[point_pair[0]][1], frame_landmarks[point_pair[1]][1]],
        #                  linestyle='-', color='blue')

        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')

        self.ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        self.ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

        # This sets the limits of the X and Y axis, based on the max and min we have found before
        self.ax.set_xlim(self.x_axis_min, self.x_axis_max)
        self.ax.set_ylim(self.y_axis_min, self.y_axis_max)

print('test')

anim = Animation()
anim.run_animation()