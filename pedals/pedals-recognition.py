def get_pedal_zone(x, y, w, h):
    # Define pedal zones: CLUTCH (left), BRAKE (middle), ACCELERATOR (right)
    if x < w * (1/3):
        return 'CLUTCH'
    elif x < w * (2/3):
        return 'BRAKE'
    else:
        return 'ACCELERATOR'


# Track last detected pedal for each foot
last_pedal_left = None
last_pedal_right = None



import cv2
import numpy as np
import os
# --- Keyboard control imports ---
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Map pedals to keys (customize as needed)
PEDAL_KEY_MAP = {
    'CLUTCH': 'v',         # Example: 'a' for clutch (unchanged)
    'BRAKE': 's',          # 's' for brake
    'ACCELERATOR': 'w',    # 'w' for accelerate
}

# Track key states to avoid repeated presses
key_state_left = None
key_state_right = None


# --- CONFIG: Crop to lower body (optional, can be set to 1.0 for full frame) ---
CROP_LOWER_BODY = 1.0


# Global variable for camera selection
CAMERA_INDEX = 2svw  # Update this index to select the desired camera
# --- Main loop: Detect black sock and interpret as gear shift ---
cap = cv2.VideoCapture(CAMERA_INDEX)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    # --- Crop to lower body (optional) ---
    crop_start = int(h * (1 - CROP_LOWER_BODY))
    frame = frame[crop_start:, :, :]
    h = frame.shape[0]

    # --- Draw pedal zones ---
    pedal_labels = ['CLUTCH', 'BRAKE', 'ACCELERATOR']
    pedal_colors = [(180, 220, 255), (180, 220, 255), (180, 220, 255)]
    for i in range(3):
        x1 = int(i * w / 3)
        x2 = int((i+1) * w / 3)
        cv2.rectangle(frame, (x1, 0), (x2, h), pedal_colors[i], 2)
        cv2.putText(frame, pedal_labels[i], (x1+10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pedal_colors[i], 2)

    # --- Detect black socks using color thresholding ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.medianBlur(mask, 7)

    # Find contours (two largest = socks)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sock_centers = []
    min_area = 500
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                sock_centers.append((cx, cy, cnt))

    # Sort by x (left to right)
    sock_centers.sort(key=lambda x: x[0])

    # --- Pedal detection logic for two feet ---
    pedal_left = None
    pedal_right = None
    if len(sock_centers) >= 1:
        # Left foot (leftmost sock)
        cx, cy, cnt = sock_centers[0]
        pedal_left = get_pedal_zone(cx, cy, w, h)
        cv2.circle(frame, (cx, cy), 15, (255,0,0), -1)
        cv2.drawContours(frame, [cnt], -1, (255,0,0), 2)
    if len(sock_centers) >= 2:
        # Right foot (rightmost sock)
        cx, cy, cnt = sock_centers[-1]
        pedal_right = get_pedal_zone(cx, cy, w, h)
        cv2.circle(frame, (cx, cy), 15, (0,0,255), -1)
        cv2.drawContours(frame, [cnt], -1, (0,0,255), 2)

    # Print/log when pedal state changes

    # --- Keyboard event logic ---
    # Handle left foot
    if pedal_left != last_pedal_left:
        # Release previous key if needed
        if last_pedal_left and last_pedal_left in PEDAL_KEY_MAP:
            keyboard.release(PEDAL_KEY_MAP[last_pedal_left])
            key_state_left = None
        # Press new key if needed
        if pedal_left and pedal_left in PEDAL_KEY_MAP:
            keyboard.press(PEDAL_KEY_MAP[pedal_left])
            key_state_left = pedal_left
            print(f"Left foot pedal: {pedal_left} (key: {PEDAL_KEY_MAP[pedal_left]})")
        last_pedal_left = pedal_left

    # Handle right foot
    if pedal_right != last_pedal_right:
        # Release previous key if needed
        if last_pedal_right and last_pedal_right in PEDAL_KEY_MAP:
            keyboard.release(PEDAL_KEY_MAP[last_pedal_right])
            key_state_right = None
        # Press new key if needed
        if pedal_right and pedal_right in PEDAL_KEY_MAP:
            keyboard.press(PEDAL_KEY_MAP[pedal_right])
            key_state_right = pedal_right
            print(f"Right foot pedal: {pedal_right} (key: {PEDAL_KEY_MAP[pedal_right]})")
        last_pedal_right = pedal_right

    # Display pedal states on frame
    left_text = f"LEFT: {pedal_left if pedal_left else 'None'}"
    right_text = f"RIGHT: {pedal_right if pedal_right else 'None'}"
    cv2.putText(frame, left_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    cv2.putText(frame, right_text, (w//2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    if not sock_centers:
        cv2.putText(frame, "No sock detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)


    cv2.imshow('Pedal Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release any pressed keys on exit
for pedal in PEDAL_KEY_MAP:
    try:
        keyboard.release(PEDAL_KEY_MAP[pedal])
    except Exception:
        pass

cap.release()
cv2.destroyAllWindows()
