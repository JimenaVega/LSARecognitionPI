import cv2
import mediapipe as mp
import time

path_video = '/home/paprika/Documents/IA/DB/LSAT/test_video_1_LSAT.mp4'

cap = cv2.VideoCapture(path_video)
# cap = cv2.VideoCapture(0) # video cam number 

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, # Tracke y detecta segun el nivel de confianza de tracking (más rápido)
                      max_num_hands=2) 

mpDraw = mp.solutions.drawing_utils # Dibuo de puntos
endTime = 0
while True:
    success, img = cap.read() # img es la imagen original que vamos a mostrar
    img = cv2.resize(img, None, fx = 0.50, fy = 0.50)

    startTime = time.time()
    fps = 1 / (startTime - endTime)
 

    # Se le envia la imagen RGB al objeto de Media Pipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # Matriz con deteccion de manos

    if results.multi_hand_landmarks:
        for eachHand in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, eachHand, mpHands.HAND_CONNECTIONS)

    endTime = startTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2)



    cv2.imshow("Test image", img)
    cv2.waitKey(1)
