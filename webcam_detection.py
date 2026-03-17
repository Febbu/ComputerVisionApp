import cv2
import os
import platform
import time
from ultralytics import YOLO

try:
    import easyocr
except ImportError:
    easyocr = None


MODEL_NAME = os.environ.get("YOLO_MODEL", "yolo26m.pt")
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
DETECT_EVERY_N_FRAMES = int(os.environ.get("DETECT_EVERY_N_FRAMES", "3"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.35"))
OCR_EVERY_N_DETECTIONS = int(os.environ.get("OCR_EVERY_N_DETECTIONS", "2"))
MIN_OCR_BOX_SIZE = int(os.environ.get("MIN_OCR_BOX_SIZE", "80"))
KNOWN_BRANDS = [
    "coca-cola", "pepsi", "sprite", "fanta", "dr pepper", "monster",
    "red bull", "nestle", "evian", "dasani", "aquafina", "gatorade",
    "nike", "adidas", "apple", "samsung", "sony", "lg", "canon",
    "toyota", "honda", "ford", "bmw", "mercedes", "tesla"
]

# Load Ultralytics detection model. Official weights download automatically.
model = YOLO(MODEL_NAME)
ocr_reader = easyocr.Reader(["en"], gpu=False) if easyocr else None

# Define colors for different classes
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (255, 128, 0), (0, 255, 128), (128, 0, 255)
]


def detect_brand(crop):
    """Run OCR on a crop and map visible text to a known brand."""
    if ocr_reader is None or crop.size == 0:
        return None, ""

    text_results = ocr_reader.readtext(crop, detail=0, paragraph=True)
    detected_text = " ".join(text_results).strip().lower()
    if not detected_text:
        return None, ""

    for brand in KNOWN_BRANDS:
        if brand in detected_text:
            return brand.title(), detected_text

    return None, detected_text


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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Webcam started! Press Q to quit.")
print(f"Loaded detection model: {MODEL_NAME}")
print(f"Capture resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
print(f"Running detection every {DETECT_EVERY_N_FRAMES} frame(s)")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"OCR enabled: {ocr_reader is not None}")

frame_index = 0
last_results = None
last_inference_time = time.time()
display_fps = 0.0
ocr_pass_index = 0
box_brand_cache = {}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam")
        break

    frame_index += 1
    now = time.time()
    elapsed = now - last_inference_time
    if elapsed > 0:
        display_fps = 1.0 / elapsed
    last_inference_time = now

    if last_results is None or frame_index % DETECT_EVERY_N_FRAMES == 0:
        last_results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        ocr_pass_index += 1

    # Count detected objects
    object_counts = {}
    brand_counts = {}

    # Draw boxes and labels
    for box in last_results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)
        label = model.names[class_id]
        box_key = (label, x1, y1, x2, y2)

        brand = None
        if (ocr_reader is not None and
                (x2 - x1) >= MIN_OCR_BOX_SIZE and
                (y2 - y1) >= MIN_OCR_BOX_SIZE):
            if box_key not in box_brand_cache or ocr_pass_index % OCR_EVERY_N_DETECTIONS == 0:
                crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
                brand, _ = detect_brand(crop)
                box_brand_cache[box_key] = brand
            else:
                brand = box_brand_cache[box_key]

        # Pick color for this class
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        if brand:
            label_text = f"{label} | {brand} {confidence:.0%}"
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        else:
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

    if brand_counts:
        y_offset += 10
        cv2.putText(frame, "BRANDS",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)
        y_offset += 25
        for brand, count in brand_counts.items():
            cv2.rectangle(frame,
                          (0, y_offset - 20),
                          (panel_width, y_offset + 5),
                          (0, 0, 0), -1)
            cv2.putText(frame, f"  {brand}: {count}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
            y_offset += 30

    # Draw FPS
    cv2.putText(frame, f"Display FPS: {display_fps:.1f}",
                (frame.shape[1] - 180, 30),
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
