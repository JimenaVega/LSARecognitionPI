import cv2
import time

cam = cv2.VideoCapture(0)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cam.set(cv2.CAP_PROP_FPS, 60)

img_counter = 0
t_start = time.time()

for i in range(1000):
    ret, frame = cam.read()
    img_counter += 1
    print(f'frame number {img_counter}')

elapsed_time = time.time() - t_start
fps = int(img_counter / elapsed_time)

print(f'Elapsed_time: {elapsed_time} - FPS: {fps}')

cam.release()
