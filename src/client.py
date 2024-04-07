import cv2
import socket
import struct
import pickle

def connect_to_server():

    client_socket = None
    attempts = 1
    
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(2)
            client_socket.connect(('192.168.0.21', 65432))
            connection = client_socket.makefile('wb')

            return client_socket, connection
        
        except (ConnectionRefusedError, OSError) as e:
            print(f"Connection attempt {attempts} failed: {e}")
            attempts += 1


def reconnect(client_socket, connection):
    if client_socket:
        client_socket.close()
    if connection:
        connection.close()

    new_socket, new_connection = connect_to_server()

    return new_socket, new_connection

def get_camera():
    cam = None
    while cam is None:
        try:
            cam = cv2.VideoCapture(0)

            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 20)
        except:
            print('Could not get camera.')
            continue
    
    return cam

client_socket, connection = connect_to_server()

cam = get_camera()

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    try:
        ret, frame = cam.read()
        result, frame = cv2.imencode('.jpg', frame, encode_param)
    except:
        print("no cam disponible")
        cam.release()
        cam = get_camera()
        continue
    
    data = pickle.dumps(frame, 0)
    size = len(data)

    try:
        client_socket.sendall(struct.pack(">L", size) + data)
        print(f'{img_counter}: {size}')

    except (ConnectionError, BrokenPipeError) as e:
        print(f'Connection error {e} when trying to send frame number {img_counter}.')

        print(f'Trying to reconnect...')
        client_socket, connection = reconnect(client_socket, connection)
        print('Reconnected.')

    img_counter += 1

cam.release()