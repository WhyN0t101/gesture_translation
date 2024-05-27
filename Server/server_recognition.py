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
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z", 26: " "
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
            print(gesture_index)
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
