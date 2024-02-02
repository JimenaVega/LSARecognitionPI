import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance


def euclidean_distance(a, b):
	return distance.euclidean((a.x, a.y, a.z), (b.x, b.y, b.z))


class SLIClient:
	pose_landmarks = []
	translated_pose_landmarks = []
	normalized_pose_landmarks = []

	r_hand_landmarks = []
	translated_r_hand_landmarks = []

	l_hand_landmarks = []
	translated_l_hand_landmarks = []

	face_landmarks = []
	translated_face_landmarks = []

	show_pose_landmarks = None
	show_translated_pose_landmarks = None
	show_normalized_pose_landmarks = None

	y_axis_min = None
	y_axis_max = None
	x_axis_min = None
	x_axis_max = None
	

	video = None
	total_frames = None
	fps = None

	ax = None

	points_to_connect = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), # cara
						 (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # brazo derecho
						 (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # brazo izquierdo
						 (11, 12), (11, 23), (12, 24), (23, 24), # torso
						#  (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # pierna derecha
						#  (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # pierna izquierda
						]

	def __init__(self, video_path) -> None:
		self.video = cv2.VideoCapture(video_path)
		self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
		self.fps = int(self.video.get(cv2.CAP_PROP_FPS))

		self.y_axis_min = 0
		self.y_axis_max = 0
		self.x_axis_min = 0
		self.x_axis_max = 0

	def run_holistic(self, show_video=False, frame_divisor=2):
		frame_number = 0

		mp_drawing = mp.solutions.drawing_utils
		mp_drawing_styles = mp.solutions.drawing_styles
		mp_holistic = mp.solutions.holistic

		with mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
		model_complexity=1,
		enable_segmentation=False,
		refine_face_landmarks=False) as holistic:

			while self.video.isOpened():
				success, image = self.video.read()
				if not success:
					print("Ignoring empty camera frame.")
					# If loading a video, use 'break' instead of 'continue'.
					break
				frame_number += 1

				if frame_number % frame_divisor == 0:
					# To improve performance, optionally mark the image as not writeable to
					# pass by reference.
					image.flags.writeable = False
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
					results = holistic.process(image)

					# Draw landmark annotation on the image.
					image.flags.writeable = True
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

					# Face landmarks
					mp_drawing.draw_landmarks(
						image,
						results.face_landmarks,
						mp_holistic.FACEMESH_CONTOURS,
						landmark_drawing_spec=None,
						connection_drawing_spec=mp_drawing_styles
						.get_default_face_mesh_contours_style())

					# Pose Landmarks
					mp_drawing.draw_landmarks(
						image,
						results.pose_landmarks,
						mp_holistic.POSE_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_pose_landmarks_style())

					# Face Mesh
					mp_drawing.draw_landmarks(
						image,
						results.face_landmarks,
						mp_holistic.FACEMESH_TESSELATION,
						landmark_drawing_spec=None,
						connection_drawing_spec=mp_drawing_styles
						.get_default_face_mesh_tesselation_style())

					# Left Hand Landmarks
					mp_drawing.draw_landmarks(
						image,
						results.left_hand_landmarks,
						mp_holistic.HAND_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_hand_landmarks_style())

					# Right Hand Landmarks
					mp_drawing.draw_landmarks(
						image,
						results.right_hand_landmarks,
						mp_holistic.HAND_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_hand_landmarks_style())

					# Flip the image horizontally for a selfie-view display.
					if show_video:
						cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

					left_shoulder = results.pose_landmarks.landmark[11]
					right_shoulder = results.pose_landmarks.landmark[12]
					nose = results.pose_landmarks.landmark[0]

					shoulders_dst = euclidean_distance(left_shoulder, right_shoulder)

					frame_pose_landmarks = []
					frame_translated_pose_landmarks = []
					frame_normalized_pose_landmarks = []

					for landmark in results.pose_landmarks.landmark[0:25]:

						x_values = [landmark.x * -1, (landmark.x - nose.x) * -1, ((landmark.x - nose.x) / shoulders_dst) * -1]
						y_values = [landmark.y * -1, (landmark.y - nose.y) * -1, ((landmark.y - nose.y) / shoulders_dst) * -1]
						z_values = [landmark.z, (landmark.z - nose.z), ((landmark.z - nose.z) / shoulders_dst)]

						frame_pose_landmarks.append([x_values[0], y_values[0], z_values[0]])
						frame_translated_pose_landmarks.append([x_values[1], y_values[1], z_values[1]])
						frame_normalized_pose_landmarks.append([x_values[2], y_values[2], z_values[2]])

						for x in x_values:
							if x < self.x_axis_min:
								self.x_axis_min = x
							elif x > self.x_axis_max:
								self.x_axis_max = x

						for y in y_values:
							if y < self.y_axis_min:
								self.y_axis_min = y
							elif y > self.y_axis_max:
								self.y_axis_max = y

					self.pose_landmarks.append(frame_pose_landmarks)
					self.translated_pose_landmarks.append(frame_translated_pose_landmarks)
					self.normalized_pose_landmarks.append(frame_normalized_pose_landmarks)

					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
			self.video.release()

	def update_pose_plot(self, frame):

		# Limpiar el gráfico anterior
		self.ax.cla()

		if self.show_pose_landmarks:
			frame_pose_landmarks = self.pose_landmarks[frame]
			for landmark in frame_pose_landmarks:
				# Plotear los nuevos puntos en 2D
				self.ax.scatter(landmark[0], landmark[1], c='r', marker='o')

		if self.show_translated_pose_landmarks:
			frame_translated_pose_landmarks = self.translated_pose_landmarks[frame]
			for landmark in frame_translated_pose_landmarks:
				# Plotear los nuevos puntos en 2D
				self.ax.scatter(landmark[0], landmark[1], c='g', marker='o')

		if self.show_normalized_pose_landmarks:
			frame_normalized_pose_landmarks = self.normalized_pose_landmarks[frame]
			for landmark in frame_normalized_pose_landmarks:
				# Plotear los nuevos puntos en 2D
				self.ax.scatter(landmark[0], landmark[1], c='g', marker='o')

		# Agregar líneas entre puntos específicos
		for point_pair in self.points_to_connect:
			if self.show_pose_landmarks:
				self.ax.plot([frame_pose_landmarks[point_pair[0]][0], frame_pose_landmarks[point_pair[1]][0]],
				[frame_pose_landmarks[point_pair[0]][1], frame_pose_landmarks[point_pair[1]][1]], linestyle='-', color='blue')
			
			if self.show_translated_pose_landmarks:
				self.ax.plot([frame_translated_pose_landmarks[point_pair[0]][0], frame_translated_pose_landmarks[point_pair[1]][0]],
				[frame_translated_pose_landmarks[point_pair[0]][1], frame_translated_pose_landmarks[point_pair[1]][1]], linestyle='-', color='blue')

			if self.show_normalized_pose_landmarks:
				self.ax.plot([frame_normalized_pose_landmarks[point_pair[0]][0], frame_normalized_pose_landmarks[point_pair[1]][0]],
				[frame_normalized_pose_landmarks[point_pair[0]][1], frame_normalized_pose_landmarks[point_pair[1]][1]], linestyle='-', color='green')


		# Configurar etiquetas de los ejes
		self.ax.set_xlabel('Eje X')
		self.ax.set_ylabel('Eje Y')

		# Agregar líneas para representar los ejes x e y
		self.ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
		self.ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

		# Fijar los rangos de los ejes x e y
		self.ax.set_xlim(self.x_axis_min, self.x_axis_max)
		self.ax.set_ylim(self.y_axis_min, self.y_axis_max)

	def run_pose_animation(self, show_pose_landmarks=False, show_translated_pose_landmarks=False, show_normalized_pose_landmarks=True):
		self.show_pose_landmarks = show_pose_landmarks
		self.show_translated_pose_landmarks = show_translated_pose_landmarks
		self.show_normalized_pose_landmarks = show_normalized_pose_landmarks

		# Crear una figura
		fig, self.ax = plt.subplots()

		# Configurar el número de fotogramas (iteraciones) y la velocidad de actualización
		animation_interval = 100 / self.fps  # en milisegundos

		# Crear la animación
		animation = FuncAnimation(fig, self.update_pose_plot, frames=self.total_frames, interval=animation_interval)

		# Mostrar el gráfico animado
		plt.show()

		
video_path = r"C:\Users\alejo\OneDrive\Documents\sign_language\media\test.mp4"

sli_data = SLIClient(video_path=video_path)

sli_data.run_holistic(show_video=True, frame_divisor=1)

sli_data.run_pose_animation(show_normalized_pose_landmarks=False, show_translated_pose_landmarks=True)