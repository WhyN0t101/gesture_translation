import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

class HandRecognition:
    def __init__(self, model_path: str):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.model = tf.keras.models.load_model(model_path)
        self.labels_mapping = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
            20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
            30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: " ",
            37: "a", 38: "b", 39: "c", 40: "d", 41: "e", 42: "f", 43: "g", 44: "h", 45: "i", 46: "j",
            47: "k", 48: "l", 49: "m", 50: "n", 51: "o", 52: "p", 53: "q", 54: "r", 55: "s", 56: "t",
            57: "u", 58: "v", 59: "w", 60: "x", 61: "y", 62: "z"
        }
        self.frame_buffer = []
        self.frame_buffer_size = 10
        self.timeout_duration = 5
        self.frames_since_last_detection = 0
        self.skip_frames = 1

    def is_hand_closed(self, hand_landmarks) -> bool:
        thumb_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.02

    def process_hand_landmarks(self, hand_landmarks, frame):
        if self.is_hand_closed(hand_landmarks):
            gesture_label = "closed"
        else:
            # Convert the frame to grayscale while retaining the original BGR format
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Resize the grayscale frame to match the input shape expected by the model
            frame_resized = cv2.resize(frame_gray, (28, 28))
            # Expand the dimensions to match the input shape expected by the model
            frame_reshaped = np.expand_dims(frame_resized, axis=-1)
            # Normalize the frame
            frame_normalized = frame_reshaped / 255.0
            # Predict the gesture label
            prediction = self.model.predict(np.expand_dims(frame_normalized, axis=0))
            gesture_index = np.argmax(prediction)
            gesture_label = self.labels_mapping[gesture_index]
        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return gesture_label, frame

    def process_frame(self, frame: np.ndarray) -> tuple[str | None, np.ndarray]:
        processed_frame = frame.copy()

        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                return self.process_hand_landmarks(hand_landmarks, processed_frame)

        self.frames_since_last_detection += 1

        if self.frames_since_last_detection >= self.timeout_duration:
            self.frame_buffer = []

        if not results.multi_hand_landmarks and self.frame_buffer:
            for buffered_frame in reversed(self.frame_buffer):
                if self.frames_since_last_detection % self.skip_frames == 0:
                    results = self.hands.process(cv2.cvtColor(buffered_frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            return self.process_hand_landmarks(hand_landmarks, processed_frame)

        return "No hand detected", processed_frame
