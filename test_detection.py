import cv2
import os
from ultralytics import YOLO

MODEL_NAME = os.environ.get("YOLO_MODEL", "yolo26x.pt")

# Load Ultralytics detection model (downloads automatically if missing).
model = YOLO(MODEL_NAME)

# Run detection on a test image from the web
results = model('https://ultralytics.com/images/bus.jpg')

# Show results
results[0].show()
print(f"Loaded detection model: {MODEL_NAME}")
print("Objects detected:")
for box in results[0].boxes:
    class_id = int(box.cls)
    confidence = float(box.conf)
    label = model.names[class_id]
    print(f"  {label}: {confidence:.0%}")
