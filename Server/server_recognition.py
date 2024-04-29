import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


class HandRecognition:
    def __init__(self, model_path):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.model = load_model(model_path)
        # Define mapping between label indices and gesture labels
        self.labels_mapping = {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
            10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
            20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
            30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z", 36: " ",
            37: "a", 38: "b", 39: "c", 40: "d", 41: "e", 42: "f", 43: "g", 44: "h", 45: "i", 46: "j",
            47: "k", 48: "l", 49: "m", 50: "n", 51: "o", 52: "p", 53: "q", 54: "r", 55: "s", 56: "t",
            57: "u", 58: "v", 59: "w", 60: "x", 61: "y", 62: "z"
        }

    def is_hand_closed(self, hand_landmarks):
        thumb_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.02

    def process_frame(self, frame):
        processed_frame = frame.copy()
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if self.is_hand_closed(hand_landmarks):
                    gesture_label = "closed"
                else:
                    # Preprocess frame for model input
                    frame_resized = cv2.resize(frame, (100, 100))
                    frame_normalized = frame_resized / 255.0  # Normalize pixel values
                    frame_reshaped = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
                    # Predict gesture label using the model
                    prediction = self.model.predict(frame_reshaped)
                    gesture_index = np.argmax(prediction)  # Get the index of the highest probability
                    gesture_label = self.labels_mapping[gesture_index]  # Map index to gesture label
                # Draw landmarks on processed frame
                self.mp_drawing.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                return gesture_label, processed_frame
        return None, processed_frame
