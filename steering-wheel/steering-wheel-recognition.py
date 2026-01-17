import cv2
import mediapipe as mp
import numpy as np
import os

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Download the hand landmarker model if not present
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, MODEL_PATH)

# Create the hand landmarker
options = HandLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2
)
landmarker = HandLandmarker.create_from_options(options)

def calculate_steering_angle(hand_landmarks, w, h):
    # Use wrist (0) and index_mcp (5) to define the hand direction
    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    x1, y1 = int(wrist.x * w), int(wrist.y * h)
    x2, y2 = int(index_mcp.x * w), int(index_mcp.y * h)
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))
    # Normalize angle to [-180, 180], where 0 is straight
    return angle

# Global variable for camera selection
CAMERA_INDEX = 2  # Update this index to select the desired camera

# Main loop for steering detection
cap = cv2.VideoCapture(CAMERA_INDEX)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h, w, _ = frame.shape

    # HandLandmarker expects a numpy array (RGB image)
    from mediapipe.tasks.python.vision.core.image import Image
    mp_img = Image(image_format=1, data=rgb_frame)
    result = landmarker.detect(mp_img)

    steering_angle = None
    if result.hand_landmarks:
        # Use the first detected hand for steering
        hand_landmarks = result.hand_landmarks[0]
        # Draw landmarks
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        # Calculate steering angle
        steering_angle = calculate_steering_angle(hand_landmarks, w, h)
        # Draw wrist and index MCP
        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]
        wx, wy = int(wrist.x * w), int(wrist.y * h)
        ix, iy = int(index_mcp.x * w), int(index_mcp.y * h)
        cv2.circle(frame, (wx, wy), 8, (255, 0, 0), 2)
        cv2.circle(frame, (ix, iy), 8, (0, 255, 0), 2)
        cv2.line(frame, (wx, wy), (ix, iy), (255, 255, 0), 2)
        # Display angle
        cv2.putText(frame, f'Steering: {int(steering_angle)} deg', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    # ...existing code...

    cv2.imshow('Steering Wheel Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
