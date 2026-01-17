import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# --- Keyboard control imports ---
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Map steering to keys
STEERING_KEY_MAP = {
    'LEFT': Key.left,
    'RIGHT': Key.right,
}

# Track key state to avoid repeated presses
key_state = None

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


# --- Improved steering wheel logic ---
# Use a smoothing buffer to reduce jitter
ANGLE_SMOOTHING_WINDOW = 7
angle_buffer = deque(maxlen=ANGLE_SMOOTHING_WINDOW)

def get_hand_anchor_point(landmarks, w, h):
    # Use wrist (0) and average of MCPs (5, 9, 13, 17) for a robust anchor
    # This works for both open and closed hands
    indices = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in indices]
    ys = [landmarks[i].y for i in indices]
    anchor_x = np.mean(xs) * w
    anchor_y = np.mean(ys) * h
    return int(anchor_x), int(anchor_y)

def calculate_steering_angle_two_hands(hands_landmarks, w, h):
    # Use robust anchor points for both hands (wrist + MCPs)
    if len(hands_landmarks) < 2:
        return None
    # Sort hands by x position to ensure left/right consistency
    hands_sorted = sorted(hands_landmarks, key=lambda lm: get_hand_anchor_point(lm, w, h)[0])
    left_hand = hands_sorted[0]
    right_hand = hands_sorted[1]
    lx, ly = get_hand_anchor_point(left_hand, w, h)
    rx, ry = get_hand_anchor_point(right_hand, w, h)
    dx = rx - lx
    dy = ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


# --- Steering to key logic ---


# --- Steering logic for two hands (steering wheel simulation) ---

# --- Improved dead zone and smoothing logic ---
STEERING_DEADZONE = 18  # degrees, neutral zone around center
STEERING_HYSTERESIS = 6  # degrees, to avoid flicker

def get_smoothed_angle(new_angle):
    if new_angle is not None:
        angle_buffer.append(new_angle)
    if len(angle_buffer) == 0:
        return None
    # Use median for robustness to outliers
    return float(np.median(angle_buffer))

def get_steering_key_two_hands(smoothed_angle, prev_key):
    if smoothed_angle is None:
        return None
    # Hysteresis: only switch to center if angle is close to 0
    if prev_key == 'LEFT':
        if smoothed_angle > -STEERING_DEADZONE + STEERING_HYSTERESIS:
            return None
        else:
            return 'LEFT'
    elif prev_key == 'RIGHT':
        if smoothed_angle < STEERING_DEADZONE - STEERING_HYSTERESIS:
            return None
        else:
            return 'RIGHT'
    else:
        if smoothed_angle < -STEERING_DEADZONE:
            return 'LEFT'
        elif smoothed_angle > STEERING_DEADZONE:
            return 'RIGHT'
        else:
            return None


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
    smoothed_angle = None
    steering_key = None
    if result.hand_landmarks and len(result.hand_landmarks) >= 2:
        # Use both hands for steering wheel simulation
        hand_landmarks_1 = result.hand_landmarks[0]
        hand_landmarks_2 = result.hand_landmarks[1]
        # Draw landmarks for both hands
        for hand_landmarks in [hand_landmarks_1, hand_landmarks_2]:
            # Draw anchor point (wrist + MCPs average)
            anchor_x, anchor_y = get_hand_anchor_point(hand_landmarks, w, h)
            cv2.circle(frame, (anchor_x, anchor_y), 7, (0, 255, 0), -1)
            # Optionally, draw all MCPs and wrist for debug
            for i in [0, 5, 9, 13, 17]:
                x, y = int(hand_landmarks[i].x * w), int(hand_landmarks[i].y * h)
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        # Draw line between anchor points
        ax1, ay1 = get_hand_anchor_point(hand_landmarks_1, w, h)
        ax2, ay2 = get_hand_anchor_point(hand_landmarks_2, w, h)
        cv2.line(frame, (ax1, ay1), (ax2, ay2), (255, 255, 0), 3)
        # Calculate steering angle between anchor points
        steering_angle = calculate_steering_angle_two_hands([hand_landmarks_1, hand_landmarks_2], w, h)
        smoothed_angle = get_smoothed_angle(steering_angle)
        steering_key = get_steering_key_two_hands(smoothed_angle, key_state)
        # Display angle
        if smoothed_angle is not None:
            cv2.putText(frame, f'Steering: {int(smoothed_angle)} deg', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)


    # --- Improved keyboard event logic for steering ---
    if steering_key != key_state:
        # Release previous key if needed
        if key_state and key_state in STEERING_KEY_MAP:
            try:
                keyboard.release(STEERING_KEY_MAP[key_state])
            except Exception:
                pass
        # Press new key if needed
        if steering_key and steering_key in STEERING_KEY_MAP:
            try:
                keyboard.press(STEERING_KEY_MAP[steering_key])
            except Exception:
                pass
        key_state = steering_key
    # If holding left/right, keep key pressed (for games that require continuous press)
    elif steering_key in STEERING_KEY_MAP and key_state == steering_key:
        try:
            keyboard.press(STEERING_KEY_MAP[steering_key])
        except Exception:
            pass

    # ...existing code...

    cv2.imshow('Steering Wheel Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release any pressed keys on exit
for k in STEERING_KEY_MAP:
    try:
        keyboard.release(STEERING_KEY_MAP[k])
    except Exception:
        pass

cap.release()
cv2.destroyAllWindows()
