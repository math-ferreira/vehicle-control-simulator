import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python open-camera.py <camera_index>")
    sys.exit(1)

try:
    cam_index = int(sys.argv[1])
except ValueError:
    print("Camera index must be an integer.")
    sys.exit(1)

cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"Failed to open camera at index {cam_index}.")
    sys.exit(1)

print(f"Camera {cam_index} opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.imshow(f'Camera {cam_index}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
