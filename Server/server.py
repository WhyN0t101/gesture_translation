import pickle
import socket
import struct
from concurrent.futures import ThreadPoolExecutor
from server_recognition import HandRecognition


class Server:
    """Server class for handling client connections and processing frames."""

    def __init__(self, host, port, model_path, max_clients=5, frame_buffer_size=10, timeout_duration=5, skip_frames=1):
        """
        Initialize the server with the given parameters.

        Args:
            host (str): The host address to bind the server socket to.
            port (int): The port number to bind the server socket to.
            model_path (str): The path to the gesture recognition model file.
            max_clients (int, optional): The maximum number of clients that can be connected simultaneously.
            frame_buffer_size (int, optional): The size of the frame buffer for each client. Defaults to 10.
            timeout_duration (int, optional): The timeout duration for receiving data from a client. Defaults to 5.
            skip_frames (int, optional): The number of frames to skip between processing. Defaults to 1.
        """
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
        """Start the server and begin listening for client connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
        """Handle a client connection and process frames."""
        with conn:
            try:
                conn.settimeout(self.timeout_duration)

                while True:
                    try:
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
                        break
                    except Exception as e:
                        print(f"Error handling client: {e}")
                        break

            except socket.timeout:
                print("Client connection timed out.")
                conn.close()

    def stop(self):
        """Stop the server and close the server socket."""
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
