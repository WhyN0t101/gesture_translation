import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pickle
from client_socket import ClientSocket
import threading
import struct

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
        send_thread = threading.Thread(target=self.send_image_continuously)
        send_thread.daemon = True  # Daemonize the thread to close it when the main thread exits
        send_thread.start()

        # Start updating the camera feed
        self.update_camera()

        # Bind the destroy event to close the socket connection
        self.protocol("WM_DELETE_WINDOW", self.destroy)

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
                # Send size of the data first
                self.client_socket.send(struct.pack("!I", len(encoded_frame)))
                # Send encoded frame over socket
                self.client_socket.send(encoded_frame)
                # Receive the sign from the server
                sign = self.client_socket.recv(4096)  # Assuming the sign is sent as a string with max length 4096
                # Process the received sign (gesture_label)
                self.process_received_sign(sign)
                # Uncomment the line below if you want to add a delay between sending frames
                # time.sleep(0.1)

    def destroy(self):
        # Close the socket connection
        self.client_socket.close()
        # Release the camera
        self.cap.release()
        # Call the superclass destroy method
        super().destroy()

    def recognition_mode(self):
        self.mode_var.set("Recognition")

    def translation_mode(self):
        self.mode_var.set("Translation")

    def process_received_sign(self, sign):
        print("Received sign:", sign)

if __name__ == "__main__":
    app = App()
    print("Client connected to the server.")
    app.mainloop()
