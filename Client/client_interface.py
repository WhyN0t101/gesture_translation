import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import threading
import struct
from client_socket import ClientSocket
import mediapipe as mp
import time


class App(tk.Tk):
    """Main application class for the client-side GUI."""

    TITLE = "Gesture Translation and Recognition"
    STYLES = {
        "button": {
            "font": ("Helvetica", 14),
            "padding": 10
        }
    }

    def __init__(self, *args, **kwargs):
        """Initialize the application."""
        super().__init__(*args, **kwargs)

        self.running = True
        self.current_mode = Mode.RECOGNITION
        self.server_connected = False
        self.title(self.TITLE)
        self.geometry("600x640")
        self.configure(bg="darkgrey")

        self.main_frame = tk.Frame(self, bg="darkgrey")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.image_frame = tk.Frame(self.main_frame, bg="white")
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.main_frame, bg="lightgrey")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.camera_label = tk.Label(self.image_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        self.gesture_text = tk.StringVar()
        self.gesture_label = ttk.Label(self.button_frame, textvariable=self.gesture_text)
        self.gesture_label.pack(side=tk.TOP, pady=10)

        self.recognition_button = ttk.Button(self.button_frame, text="Recognition Mode", command=self.recognition_mode,
                                             style="Large.TButton")
        self.recognition_button.pack(side=tk.LEFT, padx=20, pady=10, expand=True)

        self.translation_button = ttk.Button(self.button_frame, text="Translation Mode", command=self.translation_mode,
                                             style="Large.TButton")
        self.translation_button.pack(side=tk.RIGHT, padx=20, pady=10, expand=True)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            self.destroy()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.camera_lock = threading.Lock()

        self.update_camera()

        self.connect_to_server_thread = threading.Thread(target=self.connect_to_server)
        self.connect_to_server_thread.start()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_camera(self):
        """Update the camera feed."""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                try:
                    with self.camera_lock:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        imgtk = ImageTk.PhotoImage(image=img)
                        self.camera_label.imgtk = imgtk
                        self.camera_label.config(image=imgtk)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to update camera: {e}")
            self.after(33, self.update_camera)

    def connect_to_server(self):
        """Continuously attempt to connect to the server."""
        while self.running:
            if not self.server_connected:
                try:
                    self.client_socket = ClientSocket()
                    if self.client_socket.is_socket_open():
                        print("Client connected to the server.")
                        self.server_connected = True
                        self.send_thread = threading.Thread(target=self.send_image_continuously)
                        self.send_thread.start()
                except Exception as e:
                    print(f"Error connecting to server: {e}")
                    self.server_connected = False
            time.sleep(5)  # Wait before trying to reconnect

    def send_image_continuously(self):
        """Send camera frames continuously to the server."""
        while self.running and self.server_connected:
            try:
                ret, frame = self.cap.read()
                if ret:
                    cropped_frame = self.crop_hand_region(frame)
                    if cropped_frame is not None:
                        encoded_frame = pickle.dumps(cropped_frame)
                        self.client_socket.send(struct.pack("!I", len(encoded_frame)))
                        self.client_socket.send(encoded_frame)
                    else:
                        self.client_socket.send(struct.pack("!I", 0))
                        self.client_socket.send(b'')
                    sign = self.client_socket.recv(4096)
                    self.process_received_sign(sign)
                else:
                    self.client_socket.send(struct.pack("!I", 0))
                    self.client_socket.send(b'')
                    sign = self.client_socket.recv(4096)
                    self.process_received_sign(sign)
            except (ConnectionResetError, ConnectionAbortedError) as e:
                print(f"Connection error: {e}")
                self.server_connected = False
                self.client_socket.close()
            except Exception as e:
                if self.running:
                    messagebox.showerror("Error", f"Unexpected error: {e}")
                    self.on_close()

    def crop_hand_region(self, frame):
        """Detect hand landmarks and crop the hand region from the frame."""
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                h, w, _ = frame.shape
                x_min = int(x_min * w)
                x_max = int(x_max * w)
                y_min = int(y_min * h)
                y_max = int(y_max * h)
                return frame[y_min:y_max, x_min:x_max]
        return None

    def on_close(self):
        """Handle window close event."""
        self.running = False
        if self.server_connected:
            self.send_thread.join()
            self.client_socket.close()
        self.cap.release()
        self.destroy()

    def recognition_mode(self):
        """Switch to recognition mode."""
        self.current_mode = Mode.RECOGNITION

    def translation_mode(self):
        """Switch to translation mode."""
        self.current_mode = Mode.TRANSLATION

    def process_received_sign(self, sign):
        """Process the received sign."""
        if sign:
            self.gesture_text.set(sign.decode())
        else:
            self.gesture_text.set("No sign recognized")


class Mode:
    """Enum-like class to represent application modes."""
    RECOGNITION = 1
    TRANSLATION = 2


if __name__ == "__main__":
    app = App()
    app.mainloop()
