import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


class HandRecognition:
    def __init__(self, model_path: str):
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
        self.frame_buffer = []
        self.frame_buffer_size = 10
        self.timeout_duration = 5
        self.frames_since_last_detection = 0

    def is_hand_closed(self, hand_landmarks) -> bool:
        """
        Check if the hand is closed.

        :param hand_landmarks: The hand landmarks.
        :return: True if the hand is closed, False otherwise.
        """
        thumb_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y])
        index_tip = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                              hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance < 0.02

    def process_frame(self, frame: np.ndarray) -> tuple[str | None, np.ndarray]:
        """
        Process a frame and recognize hand gestures.

        :param frame: The frame to process.
        :return: A tuple containing the recognized gesture label and the processed frame.
        """
        processed_frame = frame.copy()
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Update frame buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if self.is_hand_closed(hand_landmarks):
                    gesture_label = "closed"
                else:
                    frame_resized = cv2.resize(frame, (100, 100))
                    frame_normalized = frame_resized / 255.0
                    frame_reshaped = np.expand_dims(frame_normalized, axis=0)
                    prediction = self.model.predict(frame_reshaped)
                    gesture_index = np.argmax(prediction)
                    gesture_label = self.labels_mapping[gesture_index]
                self.mp_drawing.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Reset frames since last detection
                self.frames_since_last_detection = 0
                return gesture_label, processed_frame

        # Increment frames since last detection
        self.frames_since_last_detection += 1

        # Check if hand has been missing for too long
        if self.frames_since_last_detection >= self.timeout_duration:
            # Clear frame buffer
            self.frame_buffer = []

        # Process frames in buffer if hand is missing
        if not results.multi_hand_landmarks and self.frame_buffer:
            for buffered_frame in reversed(self.frame_buffer):
                results = self.hands.process(cv2.cvtColor(buffered_frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        if self.is_hand_closed(hand_landmarks):
                            gesture_label = "closed"
                        else:
                            frame_resized = cv2.resize(buffered_frame, (100, 100))
                            frame_normalized = frame_resized / 255.0
                            frame_reshaped = np.expand_dims(frame_normalized, axis=0)
                            prediction = self.model.predict(frame_reshaped)
                            gesture_index = np.argmax(prediction)
                            gesture_label = self.labels_mapping[gesture_index]
                        self.mp_drawing.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        return gesture_label, processed_frame

        return None, processed_frame
