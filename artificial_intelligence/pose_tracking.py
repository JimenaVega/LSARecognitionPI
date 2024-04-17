import cv2
import mediapipe as mp

VideoPath = '/home/paprika/Documents/IA/DB/LSAT/test_video_1_LSAT.mp4'

cap = cv2.VideoCapture(VideoPath)

mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False) # Se reduce a 25 landmarks

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.resize(img, None, fx = 0.50, fy = 0.50)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # La imagen esta en BGR pero la libreria utiliza RGB
   

    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS) # Dots and connections



    cv2.imshow("Poses", img)
    cv2.waitKey(10)