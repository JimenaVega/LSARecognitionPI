import cv2
import time

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 15)

img_counter = 0
t_start = time.time()

for i in range(1000):
    ret, frame = cam.read()
    img_counter += 1

elapsed_time = time.time() - t_start
fps = int(img_counter / elapsed_time)

cam.release()
