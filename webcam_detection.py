import cv2
import os
import platform
from ultralytics import YOLO


MODEL_NAME = os.environ.get("YOLO_MODEL", "yolo26x.pt")

# Load Ultralytics detection model. Official weights download automatically.
model = YOLO(MODEL_NAME)

# Define colors for different classes
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (255, 128, 0), (0, 255, 128), (128, 0, 255)
]


def open_camera(max_index=5):
    """Open the first working camera with platform-specific backends."""
    system = platform.system()

    if system == 'Darwin':
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif system == 'Windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        for cam_index in range(max_index + 1):
            cap = cv2.VideoCapture(cam_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ret, _ = cap.read()
            if ret:
                print(f"Using camera index {cam_index} with backend {backend}")
                return cap

            cap.release()

    return None


# Open webcam
cap = open_camera(max_index=5)
if cap is None:
    print("Could not open any camera. Check camera permissions and camera index.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam started! Press Q to quit.")
print(f"Loaded detection model: {MODEL_NAME}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam")
        break

    # Run YOLO detection
    results = model(frame, verbose=False, conf=0.5)

    # Count detected objects
    object_counts = {}

    # Draw boxes and labels
    for box in results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)
        label = model.names[class_id]

        # Pick color for this class
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{label} {confidence:.0%}"
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame,
                      (x1, y1 - text_h - 10),
                      (x1 + text_w + 10, y1),
                      color, -1)

        # Draw label text
        cv2.putText(frame, label_text,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        # Count objects
        object_counts[label] = object_counts.get(label, 0) + 1

    # Draw stats panel on left side
    panel_width = 220
    cv2.rectangle(frame, (0, 0), (panel_width, 30), (0, 0, 0), -1)
    cv2.putText(frame, "DETECTED OBJECTS",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    y_offset = 50
    for obj, count in object_counts.items():
        cv2.rectangle(frame,
                      (0, y_offset - 20),
                      (panel_width, y_offset + 5),
                      (0, 0, 0), -1)
        cv2.putText(frame, f"  {obj}: {count}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        y_offset += 30

    # Draw FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (frame.shape[1] - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Draw total object count
    total = sum(object_counts.values())
    cv2.putText(frame, f"Total: {total} objects",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")
