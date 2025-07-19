import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("gesture_model.h5")
IMG_SIZE = 150 
GESTURES = ['palm', 'L', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# Map gestures to actions
gesture_to_key = {
    'palm': 'space',
    'fist': 'space',
    'thumb': 'volumeup',
    'down': 'volumedown',
    'L': 'right',
    'c': 'left'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
last_gesture = None
cooldown = 20
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for ROI
            h, w, _ = frame.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp to image bounds
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=(0, -1))  # Shape: (1, 64, 64, 1)

            prediction = model.predict(roi_input, verbose=0)
            confidence = np.max(prediction)
            if confidence > 0.85:
                gesture = GESTURES[np.argmax(prediction)]
            else:
                gesture = None

    # Perform action
    if gesture and gesture != last_gesture and counter <= 0:
        action = gesture_to_key.get(gesture)
        if action:
            pyautogui.press(action)
            print(f"Gesture: {gesture} → Action: {action}")
            last_gesture = gesture
            counter = cooldown

    if counter > 0:
        counter -= 1

    cv2.putText(frame, f"Gesture: {gesture if gesture else 'None'}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
