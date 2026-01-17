def get_pedal_zone(x, y, w, h):
    # Make pedal zones more sensitive to slight movements (narrower zones)
    # You can adjust the fractions below for even more sensitivity
    if x < w * 0.28:
        return 'CLUTCH'
    elif x > w * 0.72:
        return 'ACCELERATOR'
    else:
        return 'BRAKE'

last_pedal = None
# Track last detected pedal for each foot
last_pedal_left = None
last_pedal_right = None

# Persistent display of last detected pedal positions
display_pedal_left = None
display_pedal_right = None

import cv2
import numpy as np
import mediapipe as mp
import os


# Global variable for camera selection

# --- CONFIG: Crop to lower body ---
# Fraction of frame height to keep (e.g., 0.5 = lower half only)
CROP_LOWER_BODY = 0.5

CAMERA_INDEX = 2  # Update this index to select the desired camera

# --- MediaPipe Pose setup ---
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

POSE_MODEL_PATH = 'pose_landmarker_full.task'
if not os.path.exists(POSE_MODEL_PATH):
    import urllib.request
    url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)

options = PoseLandmarkerOptions(
    base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    output_segmentation_masks=False,
    num_poses=1
)
landmarker = PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(CAMERA_INDEX)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    # Draw pedal zones at the bottom (full frame)
    cv2.rectangle(frame, (0, 0), (w//3, h), (180, 220, 255), 2)
    cv2.rectangle(frame, (w//3, 0), (2*w//3, h), (180, 220, 255), 2)
    cv2.rectangle(frame, (2*w//3, 0), (w, h), (180, 220, 255), 2)
    cv2.putText(frame, 'CLUTCH', (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2)
    # --- Crop to lower body ---
    crop_start = int(h * (1 - CROP_LOWER_BODY))
    frame = frame[crop_start:, :, :]
    h = frame.shape[0]
    cv2.putText(frame, 'BRAKE', (w//3+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2)
    cv2.putText(frame, 'ACCELERATOR', (2*w//3+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 220, 255), 2)

    # --- Improved foot/leg detection using MediaPipe Pose Landmarker ---
    from mediapipe.tasks.python.vision.core.image import Image
    mp_img = Image(image_format=1, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_img)

    pedal = None
    detected_leg = None
    pedal_left = None
    pedal_right = None
    if result.pose_landmarks:
        # --- Track specific foot landmarks for pedal control ---
        # Landmarks: 31 (RIGHT TOE), 32 (LEFT TOE)
        # After cv2.flip, 32 is LEFT TOE, 31 is RIGHT TOE
        toe_landmarks = [
            (32, 'LEFT TOE'), (31, 'RIGHT TOE')
        ]
        pedal_left = None
        pedal_right = None
        prev_display_pedal_left = display_pedal_left
        prev_display_pedal_right = display_pedal_right
        for idx, leg_label in toe_landmarks:
            lm = result.pose_landmarks[0][idx]
            x, y = int(lm.x * w), int(lm.y * h)
            pedal_label = get_pedal_zone(x, y, w, h)
            if leg_label == 'LEFT TOE':
                pedal_left = pedal_label
            elif leg_label == 'RIGHT TOE':
                pedal_right = pedal_label

        # Update persistent display and log only when pedal changes
        if pedal_left != display_pedal_left:
            display_pedal_left = pedal_left
            print(f"Pedal pressed: {display_pedal_left} (LEFT TOE)")
        if pedal_right != display_pedal_right:
            display_pedal_right = pedal_right
            print(f"Pedal pressed: {display_pedal_right} (RIGHT TOE)")

        # Draw only the requested info at the top
        left_text = f"LEFT TOE: {display_pedal_left if display_pedal_left else 'None'}"
        right_text = f"RIGHT TOE: {display_pedal_right if display_pedal_right else 'None'}"
        # Dark red and dark blue, smaller font and thickness
        cv2.putText(frame, left_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 0, 120), 2)
        cv2.putText(frame, right_text, (w//2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 0, 30), 2)

    cv2.imshow('Pedal Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
