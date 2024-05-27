import cv2
import pickle
import socket
import threading
import struct
from server_recognition import HandRecognition
from concurrent.futures import ThreadPoolExecutor

class Server:
    def __init__(self, host, port, model_path, max_clients=5, frame_buffer_size=10, timeout_duration=5, skip_frames=1):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.hand_recognition = HandRecognition(model_path)
        self.server_socket = None
        self.is_running = False
        self.max_clients = max_clients
        self.executor = ThreadPoolExecutor(max_workers=self.max_clients)
        self.frame_buffer_size = frame_buffer_size
        self.timeout_duration = timeout_duration
        self.skip_frames = skip_frames

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
                    data_size_bytes = conn.recv(4)
                    if not data_size_bytes:
                        break
                    data_size = struct.unpack("!I", data_size_bytes)[0]
                    if data_size == 0:
                        conn.sendall(b'No hand detected')
                        continue
                    data = b''
                    while len(data) < data_size:
                        packet = conn.recv(data_size - len(data))
                        if not packet:
                            break
                        data += packet
                    if len(data) < data_size:
                        break
                    if data:
                        frame = pickle.loads(data)
                        gesture_label, processed_frame = self.hand_recognition.process_frame(frame)
                        if gesture_label is not None:
                            conn.sendall(gesture_label.encode())
                        else:
                            conn.sendall(b'No gesture recognized')
                    else:
                        conn.sendall(b'No gesture recognized')

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
    HOST = '127.0.0.1'
    PORT = 12345
    MODEL_PATH = r'C:\gesture_recognition_model_with_augmentation.h5'

    server = Server(HOST, PORT, MODEL_PATH)
    server.start()
    print("Server stopped.")
