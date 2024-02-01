import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance


points_to_connect = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), # cara
					 (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # brazo derecho
					 (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # brazo izquierdo
					 (11, 12), (11, 23), (12, 24), (23, 24), # torso
					 (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # pierna derecha
					 (23, 25), (25, 27), (27, 29), (27, 31), (29, 31)] # pierna izquierda


def euclidean_distance(a, b):
	return distance.euclidean((a.x, a.y, a.z), (b.x, b.y, b.z))


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

all_pose_landmarks = []
all_translated_pose_landmarks = []

cap = cv2.VideoCapture(r"C:\Users\alejo\OneDrive\Documents\sign_language\media\test.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

with mp_holistic.Holistic(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5,
	model_complexity=1,
	enable_segmentation=False,
	refine_face_landmarks=False) as holistic:

	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			break

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


		left_shoulder = results.pose_landmarks.landmark[11]
		right_shoulder = results.pose_landmarks.landmark[12]
		nose = results.pose_landmarks.landmark[0]

		dst = euclidean_distance(left_shoulder, right_shoulder)

		frame_pose_landmarks = []
		frame_translated_pose_landmarks = []

		for landmark in results.pose_landmarks.landmark:
			frame_pose_landmarks.append([landmark.x * -1, landmark.y * -1, landmark.z])
			frame_translated_pose_landmarks.append([(landmark.x - nose.x) * -1, (landmark.y - nose.y) * -1, (landmark.z - nose.z)])

		all_pose_landmarks.append(frame_pose_landmarks)
		all_translated_pose_landmarks.append(frame_translated_pose_landmarks)
		
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	cap.release()


# Función para actualizar el gráfico en cada iteración
def update_plot(frame):

	# Limpiar el gráfico anterior
	ax.cla()

	frame_pose_landmarks = all_pose_landmarks[frame]
	frame_translated_pose_landmarks = all_translated_pose_landmarks[frame]

	for landmark in frame_pose_landmarks:

		# Plotear los nuevos puntos en 2D
		ax.scatter(landmark[0], landmark[1], c='r', marker='o')

	for landmark in frame_translated_pose_landmarks:

		# Plotear los nuevos puntos en 2D
		ax.scatter(landmark[0], landmark[1], c='g', marker='o')

	# Agregar líneas entre puntos específicos
	for point_pair in points_to_connect:
		ax.plot([frame_pose_landmarks[point_pair[0]][0], frame_pose_landmarks[point_pair[1]][0]],
		  [frame_pose_landmarks[point_pair[0]][1], frame_pose_landmarks[point_pair[1]][1]], linestyle='-', color='blue')
		
		ax.plot([frame_translated_pose_landmarks[point_pair[0]][0], frame_translated_pose_landmarks[point_pair[1]][0]],
		  [frame_translated_pose_landmarks[point_pair[0]][1], frame_translated_pose_landmarks[point_pair[1]][1]], linestyle='-', color='blue')


	# Configurar etiquetas de los ejes
	ax.set_xlabel('Eje X')
	ax.set_ylabel('Eje Y')

	# Agregar líneas para representar los ejes x e y
	ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
	ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

	# Fijar los rangos de los ejes x e y
	ax.set_xlim(-1, 1)
	ax.set_ylim(-2, 2)


# Crear una figura
fig, ax = plt.subplots()

# Configurar el número de fotogramas (iteraciones) y la velocidad de actualización
animation_interval = 1000 / fps  # en milisegundos

# Crear la animación
animation = FuncAnimation(fig, update_plot, frames=total_frames, interval=animation_interval)

# Mostrar el gráfico animado
plt.show()