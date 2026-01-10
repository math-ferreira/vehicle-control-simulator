import cv2

# 1. Load the pre-trained Haar Cascade for face/head detection
# 'cv2.data.haarcascades' points to the folder where OpenCV stores these models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Initialize the video camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Convert to grayscale (essential for Haar Cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 4. Detect heads/faces
    # scaleFactor: how much the image size is reduced at each image scale
    # minNeighbors: how many neighbors each candidate rectangle should have to retain it
    heads = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 5. Draw rectangles around detected heads
    for (x, y, w, h) in heads:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Head', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Head Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
