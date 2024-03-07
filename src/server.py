import socket
import cv2
import numpy as np

# se crea el socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# se bindea el socket a la ip local y puerto
sock.bind(('localhost', 8000))

# se pone a escuchar conexiones
sock.listen()

# se acepta la conexion
conn, addr = sock.accept()

while True:
    # se recibe el tamaño del frame
    frame_size = int.from_bytes(conn.recv(4), byteorder='big')

    # se recibe el frame
    frame_data = conn.recv(frame_size)

    # se decodifica el frame
    frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), flags=3)

    # se muestra el frame
    cv2.imshow('Video', frame)

    # se corta el bucle apretando Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# se cierra el socket
sock.close()