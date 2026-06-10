import cv2
import mediapipe as mp
import csv
import os
import time

# Gesture name input
gesture_name = input("Enter gesture name: ")

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

csv_path = f"dataset/{gesture_name}.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

sample_count = 0
max_samples = 300

# Auto capture delay
capture_delay = 0.2

last_capture_time = time.time()

print(f"\nCollecting data for gesture: {gesture_name}")
print("Move your hand slowly in different angles...")
print("Press 'q' to quit\n")

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_frame)

    landmark_list = []

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        current_time = time.time()

        # Auto save samples
        if (
            len(landmark_list) == 63 and
            current_time - last_capture_time > capture_delay
        ):

            with open(csv_path, mode='a', newline='') as f:

                writer = csv.writer(f)
                writer.writerow(landmark_list)

            sample_count += 1
            last_capture_time = current_time

            print(f"Saved sample {sample_count}")

    # Display sample count
    cv2.putText(
        frame,
        f"Samples: {sample_count}/{max_samples}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SignBridge AI - Data Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('q') or sample_count >= max_samples:
        break

cap.release()
cv2.destroyAllWindows()

print("\nData collection completed!")

