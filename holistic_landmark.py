import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance


def euclidean_distance(a, b):
	return distance.euclidean((a.x, a.y, a.z), (b.x, b.y, b.z))


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(r"C:\Users\alejo\OneDrive\Documents\sign_language\media\test.mp4")
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
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.face_landmarks,
		# 	mp_holistic.FACEMESH_CONTOURS,
		# 	landmark_drawing_spec=None,
		# 	connection_drawing_spec=mp_drawing_styles
		# 	.get_default_face_mesh_contours_style())

		# Pose Landmarks
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.pose_landmarks,
		# 	mp_holistic.POSE_CONNECTIONS,
		# 	landmark_drawing_spec=mp_drawing_styles
		# 	.get_default_pose_landmarks_style())

		# Face Mesh
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.face_landmarks,
		# 	mp_holistic.FACEMESH_TESSELATION,
		# 	landmark_drawing_spec=None,
		# 	connection_drawing_spec=mp_drawing_styles
		# 	.get_default_face_mesh_tesselation_style())

		# Left Hand Landmarks
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.left_hand_landmarks,
		# 	mp_holistic.HAND_CONNECTIONS,
		# 	landmark_drawing_spec=mp_drawing_styles
		# 	.get_default_hand_landmarks_style())
		
		# Right Hand Landmarks
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.right_hand_landmarks,
		# 	mp_holistic.HAND_CONNECTIONS,
		# 	landmark_drawing_spec=mp_drawing_styles
		# 	.get_default_hand_landmarks_style())

		# Draw specific landmarks
		for landmark in results.pose_landmarks.landmark:
			landmark_x, landmark_y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])

			cv2.circle(image, (landmark_x, landmark_y), 4, (0, 255, 0), -1)


		left_shoulder = results.pose_landmarks.landmark[11]
		right_shoulder = results.pose_landmarks.landmark[12]

		distance = euclidean_distance(left_shoulder, right_shoulder)

		
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	cap.release()