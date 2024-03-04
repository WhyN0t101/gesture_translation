import cv2
import mediapipe as mp
import numpy as np
import pyautogui

def is_hand_closed(hand_landmarks):
    # Check if the tips of the index finger and thumb are close to each other
    thumb_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x,
                          hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y])
    index_tip = np.array([hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x,
                          hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y])
    
    distance = np.linalg.norm(thumb_tip - index_tip)
    
    return distance < 0.02  # You can adjust this threshold based on your preference

def main():
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands()
    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cursor_moved = False
            click_action = False
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the center of the hand
                    hand_center = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

                    # Calculate the cursor position based on the inverted center of the hand
                    cursor_x = int((1 - hand_center[0]) * 1920)
                    cursor_y = int((1 - hand_center[1]) * 1080)

                    # Move the cursor using pyautogui
                    pyautogui.moveTo(cursor_x, cursor_y)
                    cursor_moved = True

                    # Check if the hand is closed to trigger a click action
                    if is_hand_closed(hand_landmarks):
                        click_action = True

            if not cursor_moved:
                pyautogui.moveTo(1000, 1000)  # Move the cursor to a default position

            # Perform click action if hand is closed
            if click_action:
                pyautogui.click()

            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
