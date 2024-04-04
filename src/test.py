import cv2
import time

# Check for camera availability
if not cv2.VideoCapture(0).isOpened():
    print("Error: Could not open camera.")
    exit()

cam_width = 640  # Adjust as needed
cam_height = 480  # Adjust as needed
cam_fps = 30      # Adjust as needed

cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
# cam.set(cv2.CAP_PROP_FPS, cam_fps)  # Attempt to set FPS (may not be supported)

frames_counter = 0
t_start = time.time()

# Capture frames in a loop (modify termination condition as needed)
while cv2.waitKey(1) & 0xFF != ord('q'):
    ret, frame = cam.read()

    if ret:
        cv2.imshow('ImageWindow', frame)
        print(f'Frame size: {frame.shape[1]}x{frame.shape[0]} - Cam config: {cam_width}x{cam_height}')
        frames_counter += 1

elapsed_time = time.time() - t_start
real_fps = int(frames_counter / elapsed_time)

print(f"Captured {frames_counter} frames at an approximate real FPS of {real_fps}")

cam.release()
cv2.destroyAllWindows()  # Close all OpenCV windows