import socket
import cv2
import pickle
import struct

HOST='192.168.0.21'
PORT=65432

def accept_client_connection(socket, timeout):
    connection, address = socket.accept()
    connection.settimeout(timeout)

    return connection

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn = accept_client_connection(socket=s, timeout=2)

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
frames_received = 0

while True:
    while len(data) < payload_size:
        recv_data = conn.recv(4)
        if len(recv_data) == 0:
            print('Connection lost.')
            break

        # print(f'Received data: {len(recv_data)}')
        data += recv_data

    if len(data) != 0:
        # print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        # print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        print(f'Frames received: {frames_received}')
        frames_received += 1
        cv2.imshow('ImageWindow',frame)
        cv2.waitKey(1)
    else:
        conn = accept_client_connection(socket=s, timeout=2)