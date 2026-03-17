import cv2

# Try camera indexes 0 to 5
for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index: {i}")
            cap.release()
        else:
            print(f"Index {i} opened but no frame")
    else:
        print(f"Index {i} not available")
