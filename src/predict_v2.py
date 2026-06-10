
import cv2
import mediapipe as mp
import numpy as np
import time

from collections import deque
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/sign_language_model.h5")

# Load labels
labels = np.load("models/labels.npy")

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# FPS tracking
prev_time = 0

# Prediction smoothing
prediction_history = deque(maxlen=10)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.80

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    predicted_gesture = "No Gesture"

    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_list = []

            # Extract landmarks
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            if len(landmark_list) == 63:

                landmark_array = np.array(landmark_list).reshape(1, -1)

                prediction = model.predict(
                    landmark_array,
                    verbose=0
                )

                predicted_class = np.argmax(prediction)

                confidence = np.max(prediction)

                if confidence > CONFIDENCE_THRESHOLD:

                    gesture_name = labels[predicted_class]

                    prediction_history.append(gesture_name)

                    # Most common prediction
                    predicted_gesture = max(
                        set(prediction_history),
                        key=prediction_history.count
                    )

    # FPS calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # UI background
    cv2.rectangle(frame, (0, 0), (420, 120), (0, 0, 0), -1)

    # Gesture text
    cv2.putText(
        frame,
        f"Gesture: {predicted_gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Confidence text
    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # FPS text
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("SignBridge AI v2", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()