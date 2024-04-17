import cv2
import time
import datetime
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
	face_landmarks_list = detection_result.face_landmarks
	annotated_image = np.copy(rgb_image)

	# Loop through the detected faces to visualize.
	for idx in range(len(face_landmarks_list)):
		face_landmarks = face_landmarks_list[idx]

		# Draw the face landmarks.
		face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		face_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
		])

		solutions.drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks_proto,
			connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
			landmark_drawing_spec=None,
			connection_drawing_spec=mp.solutions.drawing_styles
			.get_default_face_mesh_tesselation_style())
		solutions.drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks_proto,
			connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
			landmark_drawing_spec=None,
			connection_drawing_spec=mp.solutions.drawing_styles
			.get_default_face_mesh_contours_style())
		solutions.drawing_utils.draw_landmarks(
			image=annotated_image,
			landmark_list=face_landmarks_proto,
			connections=mp.solutions.face_mesh.FACEMESH_IRISES,
				landmark_drawing_spec=None,
				connection_drawing_spec=mp.solutions.drawing_styles
				.get_default_face_mesh_iris_connections_style())

	return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
	# Extract the face blendshapes category names and scores.
	face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
	face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
	# The blendshapes are ordered in decreasing score value.
	face_blendshapes_ranks = range(len(face_blendshapes_names))

	fig, ax = plt.subplots(figsize=(12, 12))
	bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
	ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
	ax.invert_yaxis()

	# Label each bar with values
	for score, patch in zip(face_blendshapes_scores, bar.patches):
		plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

	ax.set_xlabel('Score')
	ax.set_title("Face Blendshapes")
	plt.tight_layout()
	plt.show()

def create_face_landmarker():
	model_path = r'C:\Users\alejo\OneDrive\Documents\sign_language\face_landmarker.task'

	BaseOptions = mp.tasks.BaseOptions
	FaceLandmarker = mp.tasks.vision.FaceLandmarker
	FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
	VisionRunningMode = mp.tasks.vision.RunningMode

	# Create a face landmarker instance with the video mode:
	options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
									running_mode=VisionRunningMode.VIDEO)

	return FaceLandmarker.create_from_options(options)


face_landmarker = create_face_landmarker()

# Use OpenCV’s VideoCapture to load the input video.
video = cv2.VideoCapture(r"C:\Users\alejo\OneDrive\Documents\sign_language\media\test.mp4")

# Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
# You’ll need it to calculate the timestamp for each frame.
fps = int(video.get(cv2.CAP_PROP_FPS))
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video_length = round(total_frames/fps, 2)

frame_no = 0

start_timestamp = datetime.datetime.now()

# Loop through each frame in the video using VideoCapture#read()
while video.isOpened():
	frame_exists, frame = video.read()

	# if frame is read correctly frame_exists is True
	if frame_exists:
		# print(f"for frame : {str(frame_no)} timestamp is: {str(video.get(cv2.CAP_PROP_POS_MSEC))} type: {type(video.get(cv2.CAP_PROP_POS_MSEC))}")
		frame_no += 1
		frame = cv2.resize(frame, (1280, 720))

		if frame_no % 1 == 0:
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
	
			# cv2.imshow('Frame', frame)

			# Perform face landmarking on the provided single image.
			# The face landmarker must be created with the video mode.
			face_landmarker_result = face_landmarker.detect_for_video(mp_image, int(video.get(cv2.CAP_PROP_POS_MSEC)))

			annotated_image = draw_landmarks_on_image(frame, face_landmarker_result)
			cv2.imshow('Frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

			# plot_face_blendshapes_bar_graph(face_landmarker_result.face_blendshapes[0])
			# print(face_landmarker_result.facial_transformation_matrixes)

			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			cv2.imshow('Frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
			pass
	else:
		print("Can't receive frame (stream end?). Exiting ...")
		break

processing_duration = (datetime.datetime.now() - start_timestamp).seconds

# release the video capture object
video.release()
# Closes all the windows currently opened.
cv2.destroyAllWindows()

print(f'---------- Video ----------\nFPS: {fps}\n'
	f'Total frames: {total_frames}\n'
	f'Length: {video_length}\n'
	f'Processing duration: {processing_duration}\n'
	f'---------------------------')