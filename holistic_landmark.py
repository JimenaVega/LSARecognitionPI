import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance


def euclidean_distance(a, b):
	"""
	Calculates the distance between 2 3D points.
	"""
	return distance.euclidean((a.x, a.y, a.z), (b.x, b.y, b.z))


class SLIClient:

	pose_landmarks = [] 				# Pose landmarks of interest from holistics result (upper body, arms and head points)
	translated_pose_landmarks = []		# Same landmarks but translated, based on a reference point
	normalized_pose_landmarks = []		# Translated landmarks, but normalized. They are divided by the distances between both shoulders

	r_hand_landmarks = []
	translated_r_hand_landmarks = []

	l_hand_landmarks = []
	translated_l_hand_landmarks = []

	face_landmarks = []
	translated_face_landmarks = []

	show_pose_landmarks = None
	show_translated_pose_landmarks = None
	show_normalized_pose_landmarks = None

	pose_y_axis_min = None
	pose_y_axis_max = None
	pose_x_axis_min = None
	pose_x_axis_max = None

	hands_y_axis_min = None
	hands_y_axis_max = None
	hands_x_axis_min = None
	hands_x_axis_max = None
	

	video = None
	total_frames = None
	fps = None

	ax = None

	points_to_connect = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), # Face points
						 (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Right arm points
						 (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Left arm points
						 (11, 12), (11, 23), (12, 24), (23, 24), # Upper body points
						#  (24, 26), (26, 28), (28, 30), (28, 32), (30, 32), # Right leg points
						#  (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Left leg points
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
		frame_number = 0 # Frame counter, used for skipping frames (less processing)

		# Used for graphical purposes
		mp_drawing = mp.solutions.drawing_utils
		mp_drawing_styles = mp.solutions.drawing_styles

		# Holistic model, this is the one who process the frame and returns landmarks
		mp_holistic = mp.solutions.holistic

		with mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
		model_complexity=1,
		enable_segmentation=False,
		refine_face_landmarks=False) as holistic:

			while self.video.isOpened():
				# We read frame by frame
				success, image = self.video.read()

				# If there is no more frames to process...
				if not success:
					print("Ignoring empty camera frame.")
					# If loading a video, use 'break' instead of 'continue'
					break
				frame_number += 1

				# If there is frame to process, we take only the frames whos frame number is divisible by the frame divisor
				# So if frame divisor is 2, then we take only even numbers for processing
				if frame_number % frame_divisor == 0:
					# To improve performance, optionally mark the image as not writeable to
					# pass by reference
					image.flags.writeable = False
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

					# Here we obtain the landmarks for this frame
					results = holistic.process(image)

					# Draw landmark annotation on the image
					image.flags.writeable = True
					image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

					# Drawing face landmarks
					mp_drawing.draw_landmarks(
						image,
						results.face_landmarks,
						mp_holistic.FACEMESH_CONTOURS,
						landmark_drawing_spec=None,
						connection_drawing_spec=mp_drawing_styles
						.get_default_face_mesh_contours_style())

					# Drawing pose Landmarks
					mp_drawing.draw_landmarks(
						image,
						results.pose_landmarks,
						mp_holistic.POSE_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_pose_landmarks_style())

					# Drawing face Mesh
					mp_drawing.draw_landmarks(
						image,
						results.face_landmarks,
						mp_holistic.FACEMESH_TESSELATION,
						landmark_drawing_spec=None,
						connection_drawing_spec=mp_drawing_styles
						.get_default_face_mesh_tesselation_style())

					# Drawing left hand landmarks
					mp_drawing.draw_landmarks(
						image,
						results.left_hand_landmarks,
						mp_holistic.HAND_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_hand_landmarks_style())

					# Drawing right hand landmarks
					mp_drawing.draw_landmarks(
						image,
						results.right_hand_landmarks,
						mp_holistic.HAND_CONNECTIONS,
						landmark_drawing_spec=mp_drawing_styles
						.get_default_hand_landmarks_style())

					# Flip the image horizontally for a selfie-view display
					if show_video:
						cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

					
					# -------------------- Pose Landmarks Handling -----------------------

					# We take the shoulders, and nose points
					# The nose point is used for translation, and the shoulders for normalization
					left_shoulder = results.pose_landmarks.landmark[11]
					right_shoulder = results.pose_landmarks.landmark[12]
					nose = results.pose_landmarks.landmark[0]

					shoulders_dst = euclidean_distance(left_shoulder, right_shoulder)

					# This are the normal, translated and normalized landmarks for this frame
					frame_pose_landmarks = []
					frame_translated_pose_landmarks = []
					frame_normalized_pose_landmarks = []

					for landmark in results.pose_landmarks.landmark[0:25]:

						x_values = [landmark.x * -1,								# Raw landmark x value
				  					(landmark.x - nose.x) * -1,						# Translated landmark x value
				  					((landmark.x - nose.x) / shoulders_dst) * -1]	# Translated and normalized landmark x value
						
						y_values = [landmark.y * -1, 								# Raw landmark y value
				  					(landmark.y - nose.y) * -1, 					# Translated landmark y value
									((landmark.y - nose.y) / shoulders_dst) * -1]	# Translated and normalized landmark y value
						
						z_values = [landmark.z, 									# Same for z value
				  					(landmark.z - nose.z), 
									((landmark.z - nose.z) / shoulders_dst)]

						frame_pose_landmarks.append([x_values[0], y_values[0], z_values[0]])
						frame_translated_pose_landmarks.append([x_values[1], y_values[1], z_values[1]])
						frame_normalized_pose_landmarks.append([x_values[2], y_values[2], z_values[2]])

						# We search for the max and min X values, it is used later for the plot limits definition
						for x in x_values:
							if x < self.pose_x_axis_min:
								self.pose_x_axis_min = x
							elif x > self.pose_x_axis_max:
								self.pose_x_axis_max = x

						# We search for the max and min Y values, it is used later for the plot limits definition
						for y in y_values:
							if y < self.pose_y_axis_min:
								self.pose_y_axis_min = y
							elif y > self.pose_y_axis_max:
								self.pose_y_axis_max = y

					self.pose_landmarks.append(frame_pose_landmarks)
					self.translated_pose_landmarks.append(frame_translated_pose_landmarks)
					self.normalized_pose_landmarks.append(frame_normalized_pose_landmarks)

					
					# ------------------------- Hands Landmarks Handling -------------------------------

					right_wrist = results.right_hand_landmarks.landmark[0]
					left_wrist = results.left_hand_landmarks.landmark[0]

					# This are the normal, translated and normalized hands landmarks for this frame
					frame_hands_landmarks = []
					frame_translated_hands_landmarks = []
					frame_normalized_hands_landmarks = []

					for landmark in results.right_hand_landmarks:

						x_values = [landmark.x * -1,										# Raw landmark x value
				  					(landmark.x - right_wrist.x) * -1,						# Translated landmark x value
				  					((landmark.x - right_wrist.x) / shoulders_dst) * -1]	# Translated and normalized landmark x value
						
						y_values = [landmark.y * -1, 										# Raw landmark y value
				  					(landmark.y - right_wrist.y) * -1, 						# Translated landmark y value
									((landmark.y - right_wrist.y) / shoulders_dst) * -1]	# Translated and normalized landmark y value
						
						z_values = [landmark.z, 											# Same for z value
				  					(landmark.z - nose.z), 
									((landmark.z - nose.z) / shoulders_dst)]
						
						frame_hands_landmarks.append([x_values[0], y_values[0], z_values[0]])
						frame_translated_hands_landmarks.append([x_values[1], y_values[1], z_values[1]])
						frame_normalized_hands_landmarks.append([x_values[2], y_values[2], z_values[2]])

						# We search for the max and min X values, it is used later for the plot limits definition
						for x in x_values:
							if x < self.hands_x_axis_min:
								self.hands_x_axis_min = x
							elif x > self.hands_x_axis_max:
								self.hands_x_axis_max = x

						# We search for the max and min Y values, it is used later for the plot limits definition
						for y in y_values:
							if y < self.hands_y_axis_min:
								self.hands_y_axis_min = y
							elif y > self.hands_y_axis_max:
								self.hands_y_axis_max = y

					for landmark in results.left_hand_landmarks:

						x_values = [landmark.x * -1,										# Raw landmark x value
				  					(landmark.x - left_wrist.x) * -1,						# Translated landmark x value
				  					((landmark.x - left_wrist.x) / shoulders_dst) * -1]	# Translated and normalized landmark x value
						
						y_values = [landmark.y * -1, 										# Raw landmark y value
				  					(landmark.y - left_wrist.y) * -1, 						# Translated landmark y value
									((landmark.y - left_wrist.y) / shoulders_dst) * -1]	# Translated and normalized landmark y value
						
						z_values = [landmark.z, 											# Same for z value
				  					(landmark.z - nose.z), 
									((landmark.z - nose.z) / shoulders_dst)]
						
						frame_hands_landmarks.append([x_values[0], y_values[0], z_values[0]])
						frame_translated_hands_landmarks.append([x_values[1], y_values[1], z_values[1]])
						frame_normalized_hands_landmarks.append([x_values[2], y_values[2], z_values[2]])

						# We search for the max and min X values, it is used later for the plot limits definition
						for x in x_values:
							if x < self.hands_x_axis_min:
								self.hands_x_axis_min = x
							elif x > self.hands_x_axis_max:
								self.hands_x_axis_max = x

						# We search for the max and min Y values, it is used later for the plot limits definition
						for y in y_values:
							if y < self.hands_y_axis_min:
								self.hands_y_axis_min = y
							elif y > self.hands_y_axis_max:
								self.hands_y_axis_max = y

					self.pose_landmarks.append(frame_pose_landmarks)
					self.translated_pose_landmarks.append(frame_translated_pose_landmarks)
					self.normalized_pose_landmarks.append(frame_normalized_pose_landmarks)

					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
			self.video.release()

	def update_pose_plot(self, frame):
		"""
		This method updates the values on the plot, making the animation
		"""

		# We clean the previous plot
		self.ax.cla()

		# If the boolean atribute is True, raw pose landmarks are shown
		if self.show_pose_landmarks:
			frame_pose_landmarks = self.pose_landmarks[frame]
			for landmark in frame_pose_landmarks:
				# Plots the new 2D points
				self.ax.scatter(landmark[0], landmark[1], c='r', marker='o')

		# If the boolean atribute is True, translated pose landmarks are shown
		if self.show_translated_pose_landmarks:
			frame_translated_pose_landmarks = self.translated_pose_landmarks[frame]
			for landmark in frame_translated_pose_landmarks:
				# Plots the new 2D points
				self.ax.scatter(landmark[0], landmark[1], c='g', marker='o')

		# If the boolean atribute is True, translated and normalized pose landmarks are shown
		if self.show_normalized_pose_landmarks:
			frame_normalized_pose_landmarks = self.normalized_pose_landmarks[frame]
			for landmark in frame_normalized_pose_landmarks:
				# Plots the new 2D points
				self.ax.scatter(landmark[0], landmark[1], c='g', marker='o')

		# This draws a line between the landmarks
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


		self.ax.set_xlabel('X Axis')
		self.ax.set_ylabel('Y Axis')

		self.ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
		self.ax.axvline(0, color='black', linewidth=0.5, linestyle='--')

		# This sets the limits of the X and Y axis, based on the max and min we have found before
		self.ax.set_xlim(self.pose_x_axis_min, self.pose_x_axis_max)
		self.ax.set_ylim(self.pose_y_axis_min, self.pose_y_axis_max)

	def run_pose_animation(self, show_pose_landmarks=False, show_translated_pose_landmarks=False, show_normalized_pose_landmarks=True):
		"""
		This method runs the animation of the pose landmarks.
		The boolean params are used tho choose the landmarks that will be shown in the plot.
		"""

		self.show_pose_landmarks = show_pose_landmarks
		self.show_translated_pose_landmarks = show_translated_pose_landmarks
		self.show_normalized_pose_landmarks = show_normalized_pose_landmarks

		# Creates the plot
		fig, self.ax = plt.subplots()

		# The animation interval is based on the frames per second of the video
		animation_interval = 1000 / self.fps  # miliseconds

		# Creates the animation
		animation = FuncAnimation(fig, self.update_pose_plot, frames=self.total_frames, interval=animation_interval)

		# Show the animation
		plt.show()

		
video_path = r"C:\Users\alejo\OneDrive\Documents\sign_language\media\test.mp4"

sli_data = SLIClient(video_path=video_path)

sli_data.run_holistic(show_video=True, frame_divisor=1)

sli_data.run_pose_animation(show_normalized_pose_landmarks=False, show_translated_pose_landmarks=True)