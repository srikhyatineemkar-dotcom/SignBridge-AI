import cv2
import mediapipe as mp
import numpy as np

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

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmark_list = []

            # Extract landmarks
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

            # Convert to numpy array
            landmark_array = np.array(landmark_list).reshape(1, -1)

            # Predict
            prediction = model.predict(landmark_array, verbose=0)

            predicted_class = np.argmax(prediction)

            gesture_name = labels[predicted_class]

            confidence = np.max(prediction)

            # Display prediction
            cv2.putText(
                frame,
                f"{gesture_name} ({confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("SignBridge AI - Prediction", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()