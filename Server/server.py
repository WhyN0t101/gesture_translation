import cv2
import pickle
import socket
import threading
import struct
from server_recognition import HandRecognition

class Server:
    def __init__(self, host, port, model_path):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.hand_recognition = HandRecognition(model_path)
        self.server_socket = None
        self.is_running = False

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.is_running = True
        print(f"Server listening on {self.host}:{self.port}")

        while self.is_running:
            conn, addr = self.server_socket.accept()
            with conn:
                print(f"Connected to {addr}")

                while True:
                    # Receive size of the data first
                    data_size_bytes = conn.recv(4)
                    if not data_size_bytes:
                        break
                    data_size = struct.unpack("!I", data_size_bytes)[0]
                    # Receive encoded frame over socket
                    data = b''
                    while len(data) < data_size:
                        packet = conn.recv(data_size - len(data))
                        if not packet:
                            break
                        data += packet
                    if len(data) < data_size:
                        break
                    frame = pickle.loads(data)
                    gesture_label, processed_frame = self.hand_recognition.process_frame(frame)

                    # Send back the recognized gesture label
                    conn.sendall(gesture_label.encode())

    def stop(self):
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    HOST = '127.0.0.1'  # Change this to your server's IP address
    PORT = 12345  # Change this to the port you want to use
    MODEL_PATH = r'C:\gesture_recognition_model_with_augmentation.h5'  # Change this to the path of your hand recognition model

    server = Server(HOST, PORT, MODEL_PATH)
    server_thread = threading.Thread(target=server.start)
    server_thread.start()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        server.stop()
        server_thread.join()

    print("Server stopped.")
