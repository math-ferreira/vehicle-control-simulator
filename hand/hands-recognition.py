import cv2
import mediapipe as mp
import numpy as np

# MediaPipe v0.10+ uses mp.tasks.vision.HandLandmarker

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Download the hand landmarker model if not present
MODEL_PATH = 'hand_landmarker.task'
import os
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

# Global variable for camera selection
CAMERA_INDEX = 2  # Update this index to select the desired camera

cap = cv2.VideoCapture(CAMERA_INDEX)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # HandLandmarker expects a numpy array (RGB image)
    from mediapipe.tasks.python.vision.core.image import Image
    mp_img = Image(image_format=1, data=rgb_frame)
    result = landmarker.detect(mp_img)

    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name if result.handedness and len(result.handedness) > i else "Unknown"
            # Swap handedness label for mirrored webcam (cleaner)
            swap = {"Left": "Right", "Right": "Left"}
            display_label = swap.get(handedness, handedness)
            # Get bounding box
            h, w, _ = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{display_label} Hand', (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Draw landmarks
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
