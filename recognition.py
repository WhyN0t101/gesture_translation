import cv2
import mediapipe as mp
import numpy as np

class HandRecognition:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

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
                    # Perform action for closed hand
                    pass
                else:
                    # Perform action for open hand
                    pass
                self.mp_drawing.draw_landmarks(processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return processed_frame
