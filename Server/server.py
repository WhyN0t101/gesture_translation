import cv2
import pickle
import socket
import threading
import struct
from server_recognition import HandRecognition
from concurrent.futures import ThreadPoolExecutor


class Server:
    def __init__(self, host, port, model_path, max_clients=5):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.hand_recognition = HandRecognition(model_path)
        self.server_socket = None
        self.is_running = False
        self.max_clients = max_clients
        self.executor = ThreadPoolExecutor(max_workers=self.max_clients)

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.is_running = True
        print(f"Server listening on {self.host}:{self.port}")

        try:
            while self.is_running:
                conn, addr = self.server_socket.accept()
                print(f"Connected to {addr}")

                self.executor.submit(self.handle_client, conn)

        except Exception as e:
            print(f"Error: {e}")
            self.stop()

    def handle_client(self, conn):
        with conn:
            try:
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
                    if gesture_label is not None:
                        conn.sendall(gesture_label.encode())
                    else:
                        conn.sendall(b'')  # Send an empty byte string or some default value

            except BrokenPipeError:
                print("Client disconnected.")
            except Exception as e:
                print(f"Error handling client: {e}")

    def stop(self):
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        self.executor.shutdown()


if __name__ == "__main__":
    HOST = '127.0.0.1'  # Change this to your server's IP address
    PORT = 12345  # Change this to the port you want to use
    MODEL_PATH = r'C:\gesture_recognition_model_with_augmentation.h5'  # Change this to the path of your hand recognition model

    server = Server(HOST, PORT, MODEL_PATH)
    server.start()
    print("Server stopped.")