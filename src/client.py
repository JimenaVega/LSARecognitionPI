import socket
import cv2

# se crea el socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# se conecta al server
sock.connect(('192.168.0.21', 8000))

# se obtiene video de la cam
cap = cv2.VideoCapture(0)

while True:
    # se lee frame by frame
    ret, frame = cap.read()

    # se codifica el frame
    encoded_frame = cv2.imencode('.jpg', frame)[1].tobytes()

    # se envia el tamaño del frame
    sock.sendall(len(encoded_frame).to_bytes(4, byteorder='big'))

    # se envia el frame
    sock.sendall(encoded_frame)

    # se muestra el frame
    cv2.imshow('Video', frame)

    # se corta el bucle apretando Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# se cierra el socket
sock.close()
