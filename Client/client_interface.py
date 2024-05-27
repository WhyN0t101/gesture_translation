import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import pickle
import threading
import struct
from client_socket import ClientSocket
import mediapipe as mp

class App(tk.Tk):
    """Main application class for the client-side GUI."""

    def __init__(self, *args, **kwargs):
        """Initialize the application."""
        super().__init__(*args, **kwargs)

        self.running = True
        self.title("Gesture Translation and Recognition")
        self.geometry("800x600")
        self.configure(bg="darkgrey")

        self.camera_label = tk.Label(self, bg="white")
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        self.gesture_text = tk.StringVar()
        self.gesture_label = ttk.Label(self, textvariable=self.gesture_text)
        self.gesture_label.place(relx=0.5, rely=0.1, anchor="center")

        self.mode_var = tk.StringVar(value="Recognition")
        self.recognition_button = ttk.Button(self, text="Recognition Mode", command=self.recognition_mode)
        self.recognition_button.place(relx=0.2, rely=0.9, anchor="center")
        self.translation_button = ttk.Button(self, text="Translation Mode", command=self.translation_mode)
        self.translation_button.place(relx=0.8, rely=0.9, anchor="center")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            self.destroy()

        self.client_socket = ClientSocket()
        if not self.client_socket.is_socket_open():
            messagebox.showerror("Error", "Failed to connect to the server.")
            self.destroy()

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        self.camera_lock = threading.Lock()
        self.send_thread = threading.Thread(target=self.send_image_continuously)
        self.send_thread.daemon = True
        self.send_thread.start()
        self.update_camera()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_camera(self):
        """Update the camera feed."""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                with self.camera_lock:
                    self.camera_label.imgtk = imgtk
                    self.camera_label.config(image=imgtk)
            self.after(33, self.update_camera)

    def send_image_continuously(self):
        """Send camera frames continuously to the server."""
        try:
            while self.running:
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
            if self.running:
                messagebox.showerror("Error", f"Connection error: {e}")
                self.on_close()
        except Exception as e:
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
        self.client_socket.close()
        self.cap.release()
        self.destroy()

    def recognition_mode(self):
        """Switch to recognition mode."""
        self.mode_var.set("Recognition")

    def translation_mode(self):
        """Switch to translation mode."""
        self.mode_var.set("Translation")

    def process_received_sign(self, sign):
        """Process the received sign."""
        if sign:
            self.gesture_text.set(sign.decode())
            print(sign.decode())
        else:
            self.gesture_text.set("No sign recognized")
            print("No sign recognized")

if __name__ == "__main__":
    app = App()
    app.mainloop()
