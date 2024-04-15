import cv2
import pickle
import socket
from server_recognition import HandRecognition  # Assuming the hand recognition class is in a separate file

class Server:
    def __init__(self, host, port, model_path):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.hand_recognition = HandRecognition(model_path)

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(1)
            print(f"Server listening on {self.host}:{self.port}")

            while True:
                conn, addr = server_socket.accept()
                with conn:
                    print(f"Connected to {addr}")

                    while True:
                        data = b''
                        while True:
                            packet = conn.recv(4096)
                            if not packet:
                                break
                            data += packet
                        try:
                            frame = pickle.loads(data)
                            gesture_label, processed_frame = self.hand_recognition.process_frame(frame)
                            # Send back the recognized gesture label
                            conn.sendall(pickle.dumps(gesture_label))
                        except pickle.UnpicklingError:
                            print("Error unpickling data")

if __name__ == "__main__":
    HOST = '127.0.0.1'  # Change this to your server's IP address
    PORT = 12345  # Change this to the port you want to use
    MODEL_PATH = r'C:\Users\Tiago Pereira\Desktop\DataSetTraining\gesture_recognition_model_with_augmentation.h5'  # Change this to the path of your hand recognition model

    server = Server(HOST, PORT, MODEL_PATH)
    server.start()
