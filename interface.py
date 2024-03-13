import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from recognition import HandRecognition

class App(tk.Tk):
    def __init__(self):
        super().__init__()

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

        # Initialize camera and recognition
        self.cap = cv2.VideoCapture(0)
        self.recognition = HandRecognition()
        self.update_camera()

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            if self.mode_var.get() == "Recognition":
                frame = self.recognition.process_frame(frame)
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

    def recognition_mode(self):
        self.mode_var.set("Recognition")

    def translation_mode(self):
        self.mode_var.set("Translation")

if __name__ == "__main__":
    app = App()
    app.mainloop()
