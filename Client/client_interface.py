import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pickle
from client_socket import ClientSocket

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Configure window
        self.title("Gesture Translation and Recognition")
        self.geometry("800x600")
        self.configure(bg="darkgrey")

        # Create label for displaying camera feed
        self.camera_label = tk.Label(self, bg="white")
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")

        # Create mode buttons
        self.mode_var = tk.StringVar(value="Recognition")

        self.recognition_button = ttk.Button(self, text="Recognition Mode", command=self.recognition_mode)
        self.recognition_button.place(relx=0.2, rely=0.9, anchor="center")

        self.translation_button = ttk.Button(self, text="Translation Mode", command=self.translation_mode)
        self.translation_button.place(relx=0.8, rely=0.9, anchor="center")

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # Initialize client socket
        self.client_socket = ClientSocket()

        # Start sending images continuously
        self.send_image_continuously()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to ImageTk format
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            # Update camera label with new image
            self.camera_label.imgtk = imgtk
            self.camera_label.config(image=imgtk)
        # Repeat update at 30 fps
        self.after(33, self.update_camera)

    def send_image_continuously(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to bytes
                encoded_frame = pickle.dumps(frame)
                # Send encoded frame over socket
                self.client_socket.send(encoded_frame)
                # Uncomment the line below if you want to add a delay between sending frames
                # time.sleep(0.1)

    def recognition_mode(self):
        self.mode_var.set("Recognition")

    def translation_mode(self):
        self.mode_var.set("Translation")

if __name__ == "__main__":
    app = App()
    app.mainloop()
