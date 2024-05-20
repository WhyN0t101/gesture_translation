import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import pickle
from client_socket import ClientSocket
import threading
import struct


class App(tk.Tk):
    """Main application class for the client-side GUI."""

    def __init__(self, *args, **kwargs):
        """Initialize the application."""
        super().__init__(*args, **kwargs)

        self.running = True  # Define the running attribute
        self.title("Gesture Translation and Recognition")
        self.geometry("800x600")
        self.configure(bg="darkgrey")

        # Create label for displaying camera feed
        self.camera_label = tk.Label(self, bg="white")
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        # Gesture text label
        self.gesture_text = tk.StringVar()
        self.gesture_label = ttk.Label(self, textvariable=self.gesture_text)
        self.gesture_label.place(relx=0.5, rely=0.1, anchor="center")

        # Create mode buttons
        self.mode_var = tk.StringVar(value="Recognition")
        self.recognition_button = ttk.Button(self, text="Recognition Mode", command=self.recognition_mode)
        self.recognition_button.place(relx=0.2, rely=0.9, anchor="center")
        self.translation_button = ttk.Button(self, text="Translation Mode", command=self.translation_mode)
        self.translation_button.place(relx=0.8, rely=0.9, anchor="center")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open camera.")
            self.destroy()

        # Initialize client socket
        self.client_socket = ClientSocket()
        if not self.client_socket.is_socket_open():
            messagebox.showerror("Error", "Failed to connect to the server.")
            self.destroy()

        # Lock for synchronizing access to camera_label
        self.camera_lock = threading.Lock()

        # Start sending images continuously
        send_thread = threading.Thread(target=self.send_image_continuously)
        send_thread.daemon = True  # Daemonize the thread to close it when the main thread exits
        send_thread.start()

        # Start updating the camera feed
        self.update_camera()

        # Bind the destroy event to close the socket connection
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_camera(self):
        """Update the camera feed."""
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
                    encoded_frame = pickle.dumps(frame)
                    self.client_socket.send(struct.pack("!I", len(encoded_frame)))
                    self.client_socket.send(encoded_frame)
                    sign = self.client_socket.recv(4096)
                    self.process_received_sign(sign)
                else:
                    # Send a default or empty value to indicate no gesture was recognized
                    self.client_socket.send(struct.pack("!I", 0))
                    self.client_socket.send(b'')
        except (ConnectionResetError, ConnectionAbortedError) as e:
            messagebox.showerror("Error", f"Connection error: {e}")
            self.destroy()

    def on_close(self):
        """Handle window close event."""
        self.running = False  # Set running to False to stop the send_image_continuously thread
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
